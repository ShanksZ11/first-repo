import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. 模拟数据生成 (实际使用时请替换为读取 Excel)
# ==========================================
def load_data():
    # 模拟日期范围：20250103(基期), 20250106, 20250107, 20250108
    dates = ['20250103', '20250106', '20250107', '20250108']
    
    # 1. Market Data (行情表)
    market_data = pd.DataFrame({
        'date': dates * 2,
        'stock_code': ['000001'] * 4 + ['000002'] * 4,
        'close_price': [10.0, 10.2, 10.1, 10.3,  # 000001 价格波动
                        20.0, 20.5, 21.0, 20.8]  # 000002 价格波动
    })

    # 2. Portfolio Data (持仓表 - 日末持仓)
    # 注意：T日的归因需要 T-1日的日末持仓作为 T日的日初持仓(BOD)
    portfolio_data = pd.DataFrame({
        'date': ['20250103', '20250103', '20250106', '20250106', '20250107', '20250107', '20250108', '20250108'],
        'stock_code': ['000001', '000002'] * 4,
        'position': [1000, 500,       # 0103 EOD (即 0106 BOD)
                     1200, 400,       # 0106 EOD (发生交易)
                     1200, 400,       # 0107 EOD (无交易)
                     0, 1000]         # 0108 EOD (全卖01, 买02)
    })
    
    # 3. Benchmark Data (基准表)
    benchmark_data = pd.DataFrame({
        'date': ['20250106', '20250106', '20250107', '20250107', '20250108', '20250108'],
        'stock_code': ['000001', '000002'] * 3,
        'weight': [0.5, 0.5, 0.6, 0.4, 0.5, 0.5] # 假设基准权重每日变化
    })
    
    # 4. Trading Data (交易表)
    # 0106: 买入200股01，卖出100股02
    trading_data = pd.DataFrame({
        'date': ['20250106', '20250106', '20250108', '20250108'],
        'stock_code': ['000001', '000002', '000001', '000002'],
        'buy_amount': [200, 0, 0, 600],
        'buy_cost': [10.15, 0, 0, 20.9],
        'sell_amount': [0, 100, 1200, 0],
        'sell_cost': [0, 20.2, 10.25, 0]
    })
    
    # 5. Underlying Info (基础信息)
    info_data = pd.DataFrame({
        'stock_code': ['000001', '000002'],
        'industry': ['Bank', 'Tech']
    })
    
    return market_data, portfolio_data, benchmark_data, trading_data, info_data

# ==========================================
# 2. 每日计算核心引擎 (Daily Engine)
# ==========================================
class DailyBrinsonEngine:
    # 假设 __init__ 和 get_stock_returns 保持不变
    def __init__(self, market, portfolio, benchmark, trading, info):
        self.market = market
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.trading = trading
        self.info = info

    def get_stock_returns(self, current_date, prev_date):
        """计算个股日收益率"""
        p_curr = self.market[self.market['date'] == current_date].set_index('stock_code')['close_price']
        p_prev = self.market[self.market['date'] == prev_date].set_index('stock_code')['close_price']
        returns = (p_curr / p_prev - 1).fillna(0).rename('return')
        return returns

    def calculate_single_day(self, date, prev_date):
        """
        计算单日的Brinson结果，返回行业明细和组合汇总数据
        """
        stock_rets = self.get_stock_returns(date, prev_date)
        
        # 1. 准备数据: BOD持仓 (来自T-1日末) 和 Benchmark持仓
        port_bod = self.portfolio[self.portfolio['date'] == prev_date].copy()
        bench_curr = self.benchmark[self.benchmark['date'] == date].copy()
        
        # 关联行业信息及价格
        prices_prev = self.market[self.market['date'] == prev_date].set_index('stock_code')['close_price']
        
        port_bod = port_bod.merge(self.info, on='stock_code', how='left')
        port_bod['price_prev'] = port_bod['stock_code'].map(prices_prev)
        port_bod['mv_bod'] = port_bod['position'] * port_bod['price_prev']
        
        total_port_mv_bod = port_bod['mv_bod'].sum()
        port_bod['weight_p'] = port_bod['mv_bod'] / total_port_mv_bod if total_port_mv_bod != 0 else 0
        
        bench_curr = bench_curr.merge(self.info, on='stock_code', how='left')

        # 2. 准备宽表并计算行业层面的权重和收益率
        df_merged = pd.merge(port_bod[['stock_code', 'industry', 'weight_p']], 
                             bench_curr[['stock_code', 'industry', 'weight']], 
                             on=['stock_code', 'industry'], how='outer').fillna(0)
        df_merged.rename(columns={'weight': 'weight_b'}, inplace=True)
        df_merged['return'] = df_merged['stock_code'].map(stock_rets).fillna(0)
        
        # 个股贡献
        df_merged['ctr_p'] = df_merged['weight_p'] * df_merged['return']
        df_merged['ctr_b'] = df_merged['weight_b'] * df_merged['return']
        
        # 按行业聚合
        sector_group = df_merged.groupby('industry').agg({
            'weight_p': 'sum',
            'weight_b': 'sum',
            'ctr_p': 'sum',
            'ctr_b': 'sum'
        }).reset_index()
        
        # 计算行业收益率 (避免除以0)
        sector_group['r_p_sec'] = np.where(sector_group['weight_p']!=0, sector_group['ctr_p']/sector_group['weight_p'], 0)
        sector_group['r_b_sec'] = np.where(sector_group['weight_b']!=0, sector_group['ctr_b']/sector_group['weight_b'], 0)
        
        # 3. 核心 Brinson 计算 (行业明细)
        R_b_total = sector_group['ctr_b'].sum()
        
        # Allocation, Selection, Interaction
        sector_group['allocation'] = (sector_group['weight_p'] - sector_group['weight_b']) * (sector_group['r_b_sec'] - R_b_total)
        sector_group['selection'] = sector_group['weight_b'] * (sector_group['r_p_sec'] - sector_group['r_b_sec'])
        sector_group['interaction'] = (sector_group['weight_p'] - sector_group['weight_b']) * (sector_group['r_p_sec'] - sector_group['r_b_sec'])
        
        # 4. 组合汇总数据 (用于计算 Linking Coefficient)
        R_p_bod_total = sector_group['ctr_p'].sum()
        
        # 计算 R_port_actual 和 Total Trading Effect (逻辑同前)
        prices_curr = self.market[self.market['date'] == date].set_index('stock_code')['close_price']
        port_end = self.portfolio[self.portfolio['date'] == date].copy()
        port_end['mv_end'] = port_end['position'] * port_end['stock_code'].map(prices_curr)
        total_port_mv_end = port_end['mv_end'].sum()
        
        day_trades = self.trading[self.trading['date'] == date]
        net_flow = (day_trades['buy_amount'] * day_trades['buy_cost']).sum() - \
                   (day_trades['sell_amount'] * day_trades['sell_cost']).sum()
        
        pnl_actual = total_port_mv_end - total_port_mv_bod - net_flow
        R_p_actual = pnl_actual / total_port_mv_bod if total_port_mv_bod != 0 else 0
        
        trading_effect_total = R_p_actual - R_p_bod_total
        
        # 5. 计算行业交易效应 (比例分配)
        sector_weights = port_bod.groupby('industry')['mv_bod'].sum() / total_port_mv_bod
        sector_group['weight_p_bod'] = sector_group['industry'].map(sector_weights).fillna(0)
        sector_group['trading_effect'] = trading_effect_total * sector_group['weight_p_bod']
        
        # 6. 整合结果
        sector_group['date'] = date
        
        # 返回行业明细DF 和 组合汇总DICT
        portfolio_summary = {
            'date': date,
            'R_port_actual': R_p_actual,
            'R_bench': R_b_total,
            'trading_effect_total': trading_effect_total
        }
        
        return sector_group[['date', 'industry', 'allocation', 'selection', 'interaction', 'trading_effect']], portfolio_summary


# ==========================================
# 2. 多期 GRAP 聚合器 (修改版 - 处理行业明细)
# ==========================================
def aggregate_brinson_grap(daily_sector_df, daily_portfolio_df):
    """
    输入: 行业明细DF 和 组合汇总DF (用于计算K_t)
    输出: 行业层面的多期归因汇总结果
    """
    # 1. 计算 Linking Coefficient (K_t) - 仅基于组合汇总数据
    df_port = daily_portfolio_df.sort_values('date').reset_index(drop=True)
    n = len(df_port)
    
    r_p_series = df_port['R_port_actual'].values
    r_b_series = df_port['R_bench'].values
    
    k_factors = []
    for t in range(n):
        cum_p = np.prod(1 + r_p_series[:t])
        cum_b = np.prod(1 + r_b_series[t+1:])
        k_factors.append(cum_p * cum_b)
    
    df_port['linking_coeff'] = k_factors
    
    # 2. 合并系数到行业明细表
    df_sector = daily_sector_df.merge(
        df_port[['date', 'linking_coeff']], 
        on='date', 
        how='left'
    )
    
    # 3. 计算多期结果 (按行业分组，加权求和)
    df_sector['alloc_w'] = df_sector['allocation'] * df_sector['linking_coeff']
    df_sector['selec_w'] = df_sector['selection'] * df_sector['linking_coeff']
    df_sector['inter_w'] = df_sector['interaction'] * df_sector['linking_coeff']
    df_sector['trade_w'] = df_sector['trading_effect'] * df_sector['linking_coeff'] # 注意：这是按比例分配的交易效应
    
    final_result = df_sector.groupby('industry').agg({
        'alloc_w': 'sum',
        'selec_w': 'sum',
        'inter_w': 'sum',
        'trade_w': 'sum'
    }).reset_index()
    
    final_result.rename(columns={
        'alloc_w': 'Allocation',
        'selec_w': 'Selection',
        'inter_w': 'Interaction',
        'trade_w': 'Trading_Effect'
    }, inplace=True)
    
    # 4. 计算 Total Excess (归因结果总和)
    final_result['Total_Excess'] = (
        final_result['Allocation'] + 
        final_result['Selection'] + 
        final_result['Interaction'] + 
        final_result['Trading_Effect']
    )
    
    # 5. 加入总组合的 Cumulative Excess for comparison (可选)
    Total_Excess_Check = np.prod(1 + r_p_series) - np.prod(1 + r_b_series)
    final_result.loc[len(final_result)] = {
        'industry': 'Total Portfolio Check',
        'Allocation': final_result['Allocation'].sum(),
        'Selection': final_result['Selection'].sum(),
        'Interaction': final_result['Interaction'].sum(),
        'Trading_Effect': final_result['Trading_Effect'].sum(),
        'Total_Excess': Total_Excess_Check 
    }
    
    return final_result


# ==========================================
# 3. 执行流程演示
# ==========================================

if __name__ == "__main__":
    # --- 假设 load_data() 函数已定义 ---
    def load_data():
        # 模拟数据与之前保持一致
        dates = ['20250103', '20250106', '20250107', '20250108']
        market_data = pd.DataFrame({
            'date': dates * 2,
            'stock_code': ['000001'] * 4 + ['000002'] * 4,
            'close_price': [10.0, 10.2, 10.1, 10.3, 20.0, 20.5, 21.0, 20.8]
        })
        portfolio_data = pd.DataFrame({
            'date': ['20250103', '20250103', '20250106', '20250106', '20250107', '20250107', '20250108', '20250108'],
            'stock_code': ['000001', '000002'] * 4,
            'position': [1000, 500, 1200, 400, 1200, 400, 0, 1000]
        })
        benchmark_data = pd.DataFrame({
            'date': ['20250106', '20250106', '20250107', '20250107', '20250108', '20250108'],
            'stock_code': ['000001', '000002'] * 3,
            'weight': [0.5, 0.5, 0.6, 0.4, 0.5, 0.5]
        })
        trading_data = pd.DataFrame({
            'date': ['20250106', '20250106', '20250108', '20250108'],
            'stock_code': ['000001', '000002', '000001', '000002'],
            'buy_amount': [200, 0, 0, 600],
            'buy_cost': [10.15, 0, 0, 20.9],
            'sell_amount': [0, 100, 1200, 0],
            'sell_cost': [0, 20.2, 10.25, 0]
        })
        info_data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'industry': ['Bank', 'Tech']
        })
        return market_data, portfolio_data, benchmark_data, trading_data, info_data
    
    # 1. 加载数据
    market, portfolio, benchmark, trading, info = load_data()
    engine = DailyBrinsonEngine(market, portfolio, benchmark, trading, info)
    
    # 2. 每日计算并存储结果 (存储行业明细 DF 和 组合汇总 DF)
    daily_sector_results = []
    daily_portfolio_summary = []
    
    calc_dates = [
        ('20250106', '20250103'),
        ('20250107', '20250106'),
        ('20250108', '20250107')
    ]
    
    print("--- 每日归因计算开始 (存储行业明细) ---")
    for curr_date, prev_date in calc_dates:
        sector_df, port_dict = engine.calculate_single_day(curr_date, prev_date)
        daily_sector_results.append(sector_df)
        daily_portfolio_summary.append(port_dict)

    # 汇总每日结果
    sector_results_df = pd.concat(daily_sector_results, ignore_index=True)
    portfolio_summary_df = pd.DataFrame(daily_portfolio_summary)

    # 3. 任意多期查询 (20250106 - 20250108 整个期间)
    print("\n========== 多期 GRAP 归因报告 (行业明细) ==========")
    final_attribution = aggregate_brinson_grap(sector_results_df, portfolio_summary_df)
    
    # 格式化输出
    styled_result = final_attribution.set_index('industry').style.format('{:.4%}')
    print(styled_result.to_string())

    # 校验：Total Portfolio Check 的 Total_Excess 应该等于组合累计超额收益