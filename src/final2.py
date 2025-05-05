import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os



st.set_page_config(layout="wide", page_title="S&P 500 Portfolio Analysis Dashboard")
st.title("S&P 500 Portfolio Analysis Dashboard")

# ---- 数据加载 ----
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

stock_data = pd.read_csv(os.path.join(project_root, "data", "Constituent_Price_History.csv"))
stock_data["date"]=pd.to_datetime(stock_data["date"])
stock_data["value_lag"]=stock_data.groupby("code")["value"].shift(1)
stock_data["ret"]=stock_data["value"]/stock_data["value_lag"]-1
stock_data["date"]=pd.to_datetime(stock_data["date"])

sta_data = pd.read_excel(os.path.join(project_root, "data", "Static_Data.xlsx"))
factor_cov = pd.read_excel(os.path.join(project_root, "data", "Factor_Covariance_Matrix.xlsx"))
factor_cov.set_index('Factor', inplace=True)

# Risk-free rate data
ff3 = pd.read_csv(os.path.join(project_root, "data", "FF.csv"))
# set dt as index and rename it as date
ff3['date'] = pd.to_datetime(ff3['date'],format='%Y%m%d')
# let ff3 date are those date that only in stock_data
ff3 = ff3[ff3['date'].isin(stock_data['date'])]
rf_data = ff3[['date', 'RF']].copy()


# ---- 侧边栏输入 ----
with st.sidebar:
    selected_stocks = st.multiselect(
        "Select Stocks", options=sta_data["ticker"], help="Choose stocks to construct the portfolio"
    )
    
    # 权重模式选择
    weight_mode = st.selectbox(
        "Select Weight Mode",
        options=["Equally Weighted", "Custom Weights", "Upload CSV"],
        help="Select the weight allocation method for the portfolio"
    )
    
    # 动态权重输入
    weights = []
    if weight_mode == "Equally Weighted":
        weights = [1 / len(selected_stocks)] * len(selected_stocks) if selected_stocks else []
    elif weight_mode == "Custom Weights":
        st.write("Enter the weight for each stock:")
        for stock in selected_stocks:
            weight = st.number_input(f"Weight for {stock}", min_value=0.0, max_value=1.0, step=0.01, value=1.0 / len(selected_stocks))
            weights.append(weight)
    elif weight_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Weights File (CSV)", type=["csv"], help="CSV file must contain 'ticker' and 'weight' columns")
        if uploaded_file:
            weight_data = pd.read_csv(uploaded_file)
            # 校验文件格式
            if "ticker" not in weight_data.columns or "weight" not in weight_data.columns:
                st.error("CSV file format error. Ensure it contains 'ticker' and 'weight' columns!")
            else:
                # 校验权重总和是否为 1
                if not np.isclose(weight_data["weight"].sum(), 1.0):
                    st.error("The total weight must equal 1. Please correct and re-upload!")
                else:
                    # 自动更新已选择的股票和权重
                    selected_stocks = weight_data["ticker"].tolist()
                    weights = weight_data["weight"].tolist()
                    
                    # 自动切换到 Custom Weights 模式
                    weight_mode = "Custom Weights"
                    st.success("Upload successful! Portfolio and weights have been updated.")
    
    # 校验权重
    if selected_stocks and weights:
        if sum(weights) != 1:
            st.error("The total weight must equal 1. Please reallocate weights!")
    
    benchmark = st.selectbox("Select Benchmark", options=["SPX"])  # 单选下拉框
    end_date = st.date_input("End Date", datetime(2024, 12, 10))
    start_date = st.date_input("Start Date", datetime(2024, 1, 1))

# ---- 数据筛选 ----
mask = (stock_data["date"] >= pd.to_datetime(start_date)) & \
       (stock_data["date"] <= pd.to_datetime(end_date))
filtered_data = stock_data[mask].copy()

# ---- 计算组合收益率 ----
if selected_stocks and weights:
    stock_weights = pd.DataFrame({
        "code": selected_stocks,
        "weight": weights
    })
    
    # 计算投资组合收益率
    portfolio_data = filtered_data.merge(stock_weights, left_on="code", right_on="code", how="inner")
    portfolio_data["weighted_ret"] = portfolio_data["ret"] * portfolio_data["weight"]
    portfolio_rets = portfolio_data.groupby("date")["weighted_ret"].sum().to_frame("portfolio_ret")

    # 计算基准收益率
    benchmark_ret = filtered_data[filtered_data["code"] == benchmark].set_index("date")["ret"].to_frame("benchmark_ret")
    returns_df = pd.merge(portfolio_rets, benchmark_ret, left_index=True, right_index=True, how="outer").dropna()

    # merge with rf_data
    returns_df = pd.merge(returns_df, rf_data, on="date", how="left")
    returns_df.set_index("date", inplace=True)
    
    
    # ---- 创建选项卡 ----
    tab1, tab2 ,tab3= st.tabs(["Performance Analysis", "Risk Metrics","Factor Exposure Analysis"])

    # ---- Tab 1: 收益分析 ----
    with tab1:
        st.subheader("Portfolio Summary")
        
        # 获取投资组合概况表格
        portfolio_summary = pd.merge(stock_weights, sta_data, left_on="code", right_on="ticker", how="inner")
        
        # 显示表格
        st.dataframe(
            portfolio_summary[["ticker", "sector", "name", "weight"]].set_index("ticker"),
            use_container_width=True
        )
        # 累计收益对比图
        cum_port = (1 + returns_df["portfolio_ret"]).cumprod()
        cum_bench = (1 + returns_df["benchmark_ret"]).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_port.index, y=cum_port,
            name="Portfolio", line=dict(color='royalblue')
        ))
        fig.add_trace(go.Scatter(
            x=cum_bench.index, y=cum_bench,
            name=benchmark, line=dict(color='grey')
        ))
        fig.update_layout(
            title="Cumulative Returns Comparison",
            yaxis_title="Growth Multiple",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ----- Sector Allocation 饼状图 -----
        st.subheader("Sector Allocation")
        sector_weights = portfolio_summary.groupby("sector")["weight"].sum()

        fig = px.pie(
            sector_weights.reset_index(),
            names="sector",
            values="weight",
            title="Sector Allocation",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textinfo='percent+label', textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)
        
        # ----- 分年度表现 -----
        st.subheader("Annual Performance")
        yearly_returns = returns_df.groupby(returns_df.index.year).apply(
            lambda x: pd.Series({
                'Portfolio Return': (1 + x['portfolio_ret']).prod() - 1,
                'Benchmark Return': (1 + x['benchmark_ret']).prod() - 1,
                'Excess Return': (1 + x['portfolio_ret']).prod() / (1 + x['benchmark_ret']).prod() - 1
            })
        )
        st.dataframe(
            yearly_returns.style.format({
                'Portfolio Return': '{:.2%}',
                'Benchmark Return': '{:.2%}',
                'Excess Return': '{:.2%}'
            }),
            use_container_width=True,
        )

        # ----- 多时间范围收益分析 -----
        st.subheader("Performance Analysis")
        time_range = st.radio("Select Time Range", options=["YTD", "1Y", "3Y", "5Y"], horizontal=True)
        
        # 时间范围过滤
        end_date = pd.Timestamp(end_date)
        if time_range == "YTD":
            start_date_range = pd.Timestamp(end_date.year, 1, 1)
        elif time_range == "1Y":
            start_date_range = end_date - pd.DateOffset(years=1)
        elif time_range == "3Y":
            start_date_range = end_date - pd.DateOffset(years=3)
        elif time_range == "5Y":
            start_date_range = end_date - pd.DateOffset(years=5)

        filtered_returns = returns_df[
            (returns_df.index >= start_date_range) & (returns_df.index <= end_date)
        ]

        # 计算收益
        cum_portfolio_ret  = (1 + filtered_returns["portfolio_ret"]).prod() - 1
        cum_benchmark_ret  = (1 + filtered_returns["benchmark_ret"]).prod() - 1
        if filtered_returns.empty:
            portfolio_total_ret = 0
            benchmark_total_ret = 0
        else:
            # 计算累积收益
            cum_portfolio_ret = (1 + filtered_returns["portfolio_ret"]).cumprod() - 1
            cum_benchmark_ret = (1 + filtered_returns["benchmark_ret"]).cumprod() - 1

            # 提取最后一个值
            portfolio_total_ret = cum_portfolio_ret.iloc[-1]
            benchmark_total_ret = cum_benchmark_ret.iloc[-1]
            
        excess_return = portfolio_total_ret - benchmark_total_ret

        # 显示收益表格
        st.write(f"Return Performance ({time_range})")
        performance_table = pd.DataFrame({
            "Portfolio Return": [portfolio_total_ret],
            "Benchmark Return": [benchmark_total_ret],
            "Excess Return": [excess_return]
        })
        st.dataframe(
            performance_table.style.format({
                "Portfolio Return": "{:.2%}",
                "Benchmark Return": "{:.2%}",
                "Excess Return": "{:.2%}"
            }),
            use_container_width=True
        )

        # 绘制累积收益图
        fig = go.Figure()

        # 投资组合的累积收益
        fig.add_trace(
            go.Scatter(
                x=cum_portfolio_ret.index,
                y=cum_portfolio_ret,
                mode="lines",
                name="Portfolio",
                line=dict(color="orange", width=2),
                hovertemplate="Date: %{x}<br>Portfolio Return: %{y:.2%}<extra></extra>",
            )
        )

        # 基准的累积收益
        fig.add_trace(
            go.Scatter(
                x=cum_benchmark_ret.index,
                y=cum_benchmark_ret,
                mode="lines",
                name="Benchmark",
                line=dict(color="green", width=2),
                hovertemplate="Date: %{x}<br>Benchmark Return: %{y:.2%}<extra></extra>",
            )
        )

        # 添加阴影区域（超额收益）
        fig.add_trace(
            go.Scatter(
                x=cum_portfolio_ret.index,
                y=cum_portfolio_ret - cum_benchmark_ret,
                fill="tozeroy",
                name="Excess Return",
                line=dict(color="rgba(255, 0, 0, 0.1)", width=0),
                hovertemplate="Date: %{x}<br>Excess Return: %{y:.2%}<extra></extra>",
            )
        )

        # 更新布局
        fig.update_layout(
            title=f"Cumulative Returns ({time_range})",
            xaxis_title="Date",
            yaxis_title="Portfolio Cumulative Return",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.2,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )

        fig.update_yaxes(
            tickmode="auto",  # auto ticks
            gridcolor="lightgrey", 
        )

        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
        

    # ---- Tab 2: 风险指标 ----
    with tab2:
        # 风险指标计算
        st.subheader("Risk Metrics")
        
        # 年化指标
        ann_port_return = (1 + returns_df['portfolio_ret']).prod() ** (252/len(returns_df)) - 1
        ann_vol = returns_df['portfolio_ret'].std() * np.sqrt(252)
        
        avg_risk_free_rate = returns_df['RF'].mean()  # 平均每日无风险利率
        sharpe = (ann_port_return - avg_risk_free_rate) / ann_vol  # 使用动态无风险利率计算 Sharpe Ratio
        
        # 最大回撤
        cum_port = (1 + returns_df['portfolio_ret']).cumprod()
        peak = cum_port.cummax()
        drawdown = (cum_port - peak) / peak
        max_dd = drawdown.min()
        dd_duration = (drawdown == 0).astype(int).cumsum()
        max_dd_period = dd_duration.idxmax() - drawdown[drawdown == 0].index[-1]
        
        # VaR计算
        var_95 = returns_df['portfolio_ret'].quantile(0.05)
        cvar_95 = returns_df['portfolio_ret'][returns_df['portfolio_ret'] <= var_95].mean()
        var_99 = returns_df['portfolio_ret'].quantile(0.01)
        cvar_99 = returns_df['portfolio_ret'][returns_df['portfolio_ret'] <= var_99].mean()
        
        # 指标展示
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Annualized Return", f"{ann_port_return:.2%}")
            st.metric("Annualized Volatility", f"{ann_vol:.2%}")
            st.metric("VaR (95%)", f"{var_95:.2%}")
            st.metric("CVaR (95%)", f"{cvar_95:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Max Drawdown", 
                     f"{max_dd:.2%}",
                     help=f"Last days for: {max_dd_period.days}")
            st.metric("VaR (99%)", f"{var_99:.2%}")
            st.metric("CVaR (99%)", f"{cvar_99:.2%}")
            
        # 回撤分析图
        st.subheader("Drawdown Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tozeroy',
            line=dict(color='red'),
            name='Drawdown'
        ))
        fig.update_layout(
            title="Portfolio Drawdown Curve",
            yaxis_title="Drawdown Rate",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 收益分布
        st.subheader("Return Distribution")
        # 用户选择 VaR 置信水平
        var_confidence = st.radio(
            "Select VaR Confidence Level:",
            options=[95, 99],
            index=0,  # 默认选中 95%
            horizontal=True
        )
        var_percentile = (100 - var_confidence) / 100  # 转换为分位数
        var_value = returns_df['portfolio_ret'].quantile(var_percentile)
        
        fig = px.histogram(
            returns_df,
            x='portfolio_ret',
            nbins=50,
            title="Portfolio Daily Returns Distribution",
            labels={'portfolio_ret': 'Daily Returns'},
        )
        
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{var_confidence}% VaR",
            annotation_position="top right"
        )
    
    # 更新图像布局
        fig.update_layout(
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    with tab3:
         st.subheader("Factor Exposure Analysis")
    
    # 1. 加载因子暴露数据
         factor_exp = pd.read_excel(os.path.join(project_root, "data", "Factor_Exposures.xlsx"))
    
    # 2. 处理投资组合数据
         sel_exp = factor_exp[factor_exp["Ticker"].isin(selected_stocks)]
         sel_exp_nonull = sel_exp.dropna(axis=1, how="all")
         exp_weight = pd.merge(
         sel_exp_nonull,
         stock_weights,
         left_on="Ticker",
         right_on="code",
         how="inner"
         )
    
    # 3. 识别因子列
         non_factor_cols = ['Ticker', 'code', 'weight',"Country"]  # 根据实际数据调整
         factor_columns = [col for col in exp_weight.columns if col not in non_factor_cols]
    
    # 4. 计算组合因子暴露
         portfolio_exposure = exp_weight[factor_columns].mul(exp_weight['weight'], axis=0).sum()
    
    # 5. 用户选择因子
         selected_factors = st.multiselect(
        "Choose Factors (Null values have been filtered)",
         options=factor_columns,
         default=factor_columns[:min(5, len(factor_columns))]  # 默认最多显示前5个
         )
    
    # 6. 计算SPX500基准暴露（等权平均）
         if len(selected_factors) > 0:
            spx_exposure = factor_exp[selected_factors].mean()

        # 创建对比表格
         exposure_table = pd.DataFrame({
            "Factor": selected_factors,
            "Portfolio Exposure": portfolio_exposure[selected_factors],
            "SPX500 Exposure": spx_exposure.values,
            "Active Exposure": portfolio_exposure[selected_factors] - spx_exposure.values
             }).sort_values("Active Exposure", key=abs, ascending=False)
        
        # 7. 显示数据表格
         st.dataframe(
            exposure_table.style.format({
                "Portfolio Exposure": "{:.3f}",
                "SPX500 Exposure": "{:.3f}",
                "Active Exposure": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # 8. 可视化（组合 vs SPX）
         fig = go.Figure()
        
        # 组合暴露（柱状图）
         fig.add_trace(go.Bar(
            x=exposure_table["Factor"],
            y=exposure_table["Portfolio Exposure"],
            name='Portfolio Exposure',
            marker_color='royalblue'
         ))
        
        # SPX暴露（散点图叠加）
         fig.add_trace(go.Scatter(
            x=exposure_table["Factor"],
            y=exposure_table["SPX500 Exposure"],
            name='SPX500 Benchmark',
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='diamond'
            )
        ))
         st.plotly_chart(fig, use_container_width=True)

         st.subheader("Factor Risk Contribution Analysis")
        
        # 1. 筛选所选因子的协方差矩阵
         selected_factor_cov = factor_cov.loc[selected_factors, selected_factors]
        
        # 2. 将组合暴露转换为向量
         portfolio_vector = portfolio_exposure[selected_factors].values.reshape(-1, 1)
        
        # 3. 计算组合总风险（方差）
         portfolio_variance = portfolio_vector.T @ selected_factor_cov.values @ portfolio_vector
        
        # 4. 计算边际风险贡献
         marginal_contributions = (selected_factor_cov.values @ portfolio_vector) / np.sqrt(portfolio_variance)
        
        # 5. 计算各因子风险贡献
         risk_contributions = (portfolio_vector * marginal_contributions).flatten()
        
        # 6. 创建风险贡献表格
         risk_contribution_table = pd.DataFrame({
            "Factor": selected_factors,
            "Risk Contribution": risk_contributions,
            "Contribution ratio": risk_contributions / np.sum(np.abs(risk_contributions))  # 绝对贡献占比
         }).sort_values("Risk Contribution", key=abs, ascending=False)
        
        # 7. 显示风险贡献表格
         st.dataframe(
            risk_contribution_table.style.format({
                "Risk Contribution": "{:.4f}",
                "Contribution ratio": "{:.1%}"
            }).bar(subset=["Risk Contribution"], align='mid', color=['#d65f5f', '#5fba7d']),
            use_container_width=True,
            height=(len(selected_factors) + 1) * 35 + 3
         )
        
        # 8. 风险贡献可视化
         fig_risk = px.bar(
            risk_contribution_table,
            x="Factor",
            y="Risk Contribution",
            color="Risk Contribution",
            color_continuous_scale=["red", "lightgray", "green"],
            title="Risk Contribution of each Factors", 
            labels={"Risk Contribution": "Risk Contribution Value"}
         )
        
        # 添加参考线
         fig_risk.add_hline(y=0, line_dash="dot", line_color="black")
        
        # 更新布局
         fig_risk.update_layout(
            showlegend=False,
            yaxis_title="Risk Contribution",
            coloraxis_showscale=False,
            hovermode="x unified"
         )
        
         st.plotly_chart(fig_risk, use_container_width=True)
        
        # 9. 风险贡献占比饼图
         fig_pie = px.pie(
            risk_contribution_table,
            names="Factor",
            values="Risk Contribution",
            title="Risk Contribution Ratio",
            hole=0.4
         )
         st.plotly_chart(fig_pie, use_container_width=True)