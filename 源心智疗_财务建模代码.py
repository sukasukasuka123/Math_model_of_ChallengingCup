# -*- coding: utf-8 -*-
# 源心智疗 财务建模与蒙特卡洛仿真 + 热力图 + K线图
# ==============================================
# 本脚本主要功能：
# 定义基础业务参数与三种情景（保守 / 基础 / 激进）；
# 计算各情景下未来3年净现值（NPV）；
# 进行敏感性分析（采用率 × 单价）热力图；
# 基于Monte Carlo仿真生成年度利润“K线图”；
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 参数定义 ====
params = {
    "pilot_universities": 1,  # 第一阶段：本校试点
    "students_per_uni": 5000,  # 试点覆盖学生数
    "scenarios": {
        "campus_pilot": {  # 第一阶段：校园验证
            "adoption_rate": 0.10,  # 10%采用率
            "price_per_user_year": 30.0,  # C端订阅
            "b2b_annual_fee": 80000,  # B端年费
            "hosting_cost_per_user_month": 2.0,
            "other_opex_pct": 0.30
        },
        "regional_demo": {  # 第二阶段：区域示范
            "adoption_rate": 0.15,
            "price_per_user_year": 40.0,
            "b2b_annual_fee": 100000,
            "hosting_cost_per_user_month": 1.8,
            "other_opex_pct": 0.25
        },
        "scaling": {  # 第三阶段：规模化复制
            "adoption_rate": 0.20,
            "price_per_user_year": 50.0,
            "b2b_annual_fee": 120000,
            "hosting_cost_per_user_month": 1.5,
            "other_opex_pct": 0.20
        }
    },
    # --- 成本类参数更新 ---
    "mvp_dev_cost_rmb": 60000.0,  

    "cac_per_uni": 10000.0,  

    "gpu_monthly_cost_rmb": 800.0,  

    "ops_salary_annual_rmb": 28000.0,   

    "discount_rate": 0.12,  # 更新为12%（反映初创企业融资成本）

    "horizon_years": 3,  # 不变（短期预测周期合理）

    "growth_rate_users": 0.30  # 更新为30% CAGR（基于同类SaaS增长数据）
}

# ==== 场景财务计算（三阶段推广路径）====
def scenario_financials(params, scen_key):
    """
    输入：参数字典 + 场景键（campus_pilot / regional_demo / scaling）
    输出：该场景下的年度现金流表和3年净现值（NPV）
    """
    s = params["scenarios"][scen_key]
    
    # 根据阶段设置不同的合作高校数量
    if scen_key == "campus_pilot":
        universities = 1  # 第一阶段：本校试点
    elif scen_key == "regional_demo":
        universities = 4  # 第二阶段：3-5所高校，取中值4
    else:  # scaling
        universities = 15  # 第三阶段：10-20所高校，取中值15

    # 计算总用户与收入
    total_students = universities * params["students_per_uni"]  # 总学生数
    active_users = total_students * s["adoption_rate"]          # 活跃用户数
    
    # 收入构成：B端年费 + C端订阅收入
    b2b_revenue = s["b2b_annual_fee"] * universities           # B端年费总收入
    c_revenue = active_users * s["price_per_user_year"]        # C端订阅收入
    annual_revenue = b2b_revenue + c_revenue                   # 年总收入

    # 年度成本项
    hosting_annual = active_users * s["hosting_cost_per_user_month"] * 12.0  # 服务器成本
    gpu_annual = params["gpu_monthly_cost_rmb"] * 12.0                       # GPU成本
    ops_annual = params["ops_salary_annual_rmb"]                             # 运维成本
    cac_total = params["cac_per_uni"] * universities                         # 获客成本总额
    other_opex = s["other_opex_pct"] * annual_revenue                        # 其他运营费用

    # EBITDA与净利润
    annual_costs = hosting_annual + gpu_annual + ops_annual + other_opex
    ebitda = annual_revenue - annual_costs
    dev_amort_annual = params["mvp_dev_cost_rmb"] / params["horizon_years"]  # 开发成本摊销
    net_profit = ebitda - dev_amort_annual - cac_total / params["horizon_years"]  # 净利润

    # 构建未来3年现金流表
    cashflows = []
    users_year0 = active_users
    universities_year0 = universities
    
    for y in range(1, params["horizon_years"] + 1):
        # 逐年增长：用户数 + 合作高校数
        users_y = users_year0 * ((1 + params["growth_rate_users"]) ** (y - 1))
        
        if scen_key == "campus_pilot":
            universities_y = universities_year0  # 第一阶段保持1所
        elif scen_key == "regional_demo":
            # 第二阶段：从1所增长到4所
            universities_y = min(universities_year0 + y, 4)
        else:  # scaling
            # 第三阶段：从4所增长到15所
            universities_y = min(universities_year0 + y * 4, 15)
        
        # 收入计算
        b2b_revenue_y = s["b2b_annual_fee"] * universities_y
        c_revenue_y = users_y * s["price_per_user_year"]
        rev_y = b2b_revenue_y + c_revenue_y
        
        # 成本计算
        host_y = users_y * s["hosting_cost_per_user_month"] * 12.0
        other_y = s["other_opex_pct"] * rev_y
        costs_y = host_y + params["gpu_monthly_cost_rmb"] * 12.0 + params["ops_salary_annual_rmb"] + other_y
        
        # 利润计算
        ebitda_y = rev_y - costs_y
        dev_amort_y = params["mvp_dev_cost_rmb"] / params["horizon_years"]
        cac_amort_y = cac_total / params["horizon_years"]
        net_y = ebitda_y - dev_amort_y - cac_amort_y
        
        cashflows.append({
            "year": y,
            "universities": universities_y,
            "users": round(users_y),
            "b2b_revenue": round(b2b_revenue_y, 2),
            "c_revenue": round(c_revenue_y, 2),
            "revenue": round(rev_y, 2),
            "costs": round(costs_y, 2),
            "ebitda": round(ebitda_y, 2),
            "net_profit": round(net_y, 2)
        })

    # 计算NPV（净现值）
    dr = params["discount_rate"]
    npv = sum([cf["net_profit"] / ((1 + dr) ** cf["year"]) for cf in cashflows])
    
    return {
        "scenario": scen_key, 
        "cashflows": cashflows, 
        "npv_3yr": round(npv, 2),
        "universities": universities,
        "active_users": active_users
}

# ==== 敏感性分析热力图 ====
def sensitivity_heatmap(params):
    """
    模拟不同采用率与单价组合下的净现值（NPV）
    绘制二维热力图，展示NPV变化趋势
    """
    adoption_rates = np.linspace(0.05, 0.25, 10)   # 采用率从5%到25%
    prices = np.linspace(5, 30, 10)                # 单价从5到30美元
    npv_matrix = np.zeros((len(prices), len(adoption_rates)))

    # 遍历所有组合计算NPV
    for i, p in enumerate(prices):
        for j, a in enumerate(adoption_rates):
            tmp = params.copy()
            tmp_scen = params["scenarios"]["regional_demo"].copy()  # 使用区域示范作为基准
            tmp_scen["price_per_user_year"] = p
            tmp_scen["adoption_rate"] = a
            tmp["scenarios"] = {"sim": tmp_scen}
            res = scenario_financials(tmp, "sim")
            npv_matrix[i, j] = res["npv_3yr"]

    # 绘制热力图
    plt.figure(figsize=(7, 6))
    sns.heatmap(npv_matrix,xticklabels=np.round(adoption_rates,2), yticklabels=np.round(prices, 2),
    cmap="YlGnBu", cbar_kws={'label': 'NPV ($)'})
    plt.title("敏感性分析热力图（NPV对价格与采用率的变化）", fontproperties="SimHei")
    plt.xlabel("采用率（Adoption Rate）", fontproperties="SimHei")
    plt.ylabel("单价（Price per User per Year）", fontproperties="SimHei")
    plt.tight_layout()
    plt.savefig("sensitivity_heatmap.png", dpi=300, bbox_inches='tight')  # 导出图片
    plt.show()


# ==== K线图（年度利润分布） ====
def profit_candlestick(params, n=200):
    """
    Monte Carlo仿真：在价格、采用率、成本波动下
    模拟200次年度净利润分布，绘制“K线图”风格可视化
    """
    np.random.seed(42)
    all_results = {1: [], 2: [], 3: []}

    # Monte Carlo抽样
    for _ in range(n):
        p=params["scenarios"]["regional_demo"]["price_per_user_year"] * np.random.uniform(0.7, 1.3)
a = params["scenarios"]["regional_demo"]["adoption_rate"] * np.random.uniform(0.5, 1.5)
h=params["scenarios"]["regional_demo"]["hosting_cost_per_user_month"] * np.random.uniform(0.8, 1.2)
tmp_scen = params["scenarios"]["regional_demo"].copy()

        tmp_scen = params["scenarios"]["base"].copy()
        tmp_scen["price_per_user_year"], tmp_scen["adoption_rate"], tmp_scen["hosting_cost_per_user_month"] = p, a, h
        tmp["scenarios"] = {"sim": tmp_scen}
        cf = scenario_financials(tmp, "sim")["cashflows"]

        # 记录各年净利润
        for year_data in cf:
            all_results[year_data["year"]].append(year_data["net_profit"])

    # 绘制K线图样式的年度利润分布
    plt.figure(figsize=(7, 5))
    for y in range(1, 4):
        data = all_results[y]
        median = np.median(data)               # 中位数
        q1, q3 = np.percentile(data, [25, 75]) # 四分位数
        low, high = np.min(data), np.max(data)
        plt.plot([y, y], [low, high], color='gray', lw=2)         # 上下影线
        plt.plot([y - 0.2, y + 0.2], [q1, q1], color='blue', lw=3)# 下边界
        plt.plot([y - 0.2, y + 0.2], [q3, q3], color='blue', lw=3)# 上边界
        plt.plot(y, median, 'ro', label='中位值' if y == 1 else "")# 中位点

    plt.title("Monte Carlo 年度净利润K线图", fontproperties="SimHei")
    plt.xlabel("年份", fontproperties="SimHei")
    plt.ylabel("净利润 ($)", fontproperties="SimHei")
    plt.legend(prop={"family": "SimHei"})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("profit_candlestick.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==== 图表函数 ====
def summary_charts(params):
    # ==== 修改：情景名称改为三阶段 ====
    scenarios = ["campus_pilot", "regional_demo", "scaling"]
    all_data = []

    # 收集三种情景的年度数据
    for s in scenarios:
        res = scenario_financials(params, s)
        npv = res["npv_3yr"]
        for cf in res["cashflows"]:
            all_data.append({
                "scenario": s,
                "year": cf["year"],
                "revenue": cf["revenue"],
                "net_profit": cf["net_profit"],
                "npv": npv
            })

    df = pd.DataFrame(all_data)
    
    # ==== 修改：图表标题改为三阶段推广路径 ====
    
    # 1️⃣ 年度收入对比柱状图
    plt.figure(figsize=(7, 5))
    # 创建情景名称映射（英文->中文）
    scenario_names = {
        "campus_pilot": "校园验证",
        "regional_demo": "区域示范", 
        "scaling": "规模化复制"
    }
    # 在数据框中添加中文情景名称
    df['scenario_cn'] = df['scenario'].map(scenario_names)
    
    sns.barplot(data=df, x="year", y="revenue", hue="scenario_cn", palette="Set2")
    plt.title("三阶段推广路径下的年度收入对比", fontproperties="SimHei")  # 修改标题
    plt.xlabel("年份", fontproperties="SimHei")
    plt.ylabel("年度收入 ($)", fontproperties="SimHei")
    plt.legend(title="推广阶段", prop={"family": "SimHei"})
    plt.tight_layout()
    plt.savefig("annual_revenue_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2️⃣ 年度净利润折线图
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=df, x="year", y="net_profit", hue="scenario_cn", marker="o")
    plt.title("三阶段推广路径下的年度净利润变化", fontproperties="SimHei")  # 修改标题
    plt.xlabel("年份", fontproperties="SimHei")
    plt.ylabel("净利润 ($)", fontproperties="SimHei")
    plt.legend(title="推广阶段", prop={"family": "SimHei"})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("net_profit_trend.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3️⃣ NPV 对比条形图
    npv_df = df.groupby("scenario")["npv"].mean().reset_index()
    # 添加中文情景名称
    npv_df['scenario_cn'] = npv_df['scenario'].map(scenario_names)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(data=npv_df, x="scenario_cn", y="npv", palette="viridis")
    plt.title("三阶段推广路径的净现值（NPV）对比", fontproperties="SimHei")  # 修改标题
    plt.xlabel("推广阶段", fontproperties="SimHei")
    plt.ylabel("三年净现值 ($)", fontproperties="SimHei")
    plt.tight_layout()
    plt.savefig("npv_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4️⃣ 新增：B端与C端收入构成堆叠图
    plt.figure(figsize=(8, 6))
    
    # 收集详细的收入数据
    detailed_data = []
    for s in scenarios:
        res = scenario_financials(params, s)
        for cf in res["cashflows"]:
            detailed_data.append({
                "scenario": scenario_names[s],
                "year": cf["year"],
                "B端收入": cf["b2b_revenue"],
                "C端收入": cf["c_revenue"],
                "总收入": cf["revenue"]
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # 创建堆叠柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    years = [1, 2, 3]
    x_pos = np.arange(len(years))
    
    colors = ['#FF9999', '#66B2FF']
    
    for i, scenario in enumerate(scenario_names.values()):
        scenario_data = detailed_df[detailed_df['scenario'] == scenario]
        b2b_values = [scenario_data[scenario_data['year'] == y]['B端收入'].values[0] for y in years]
        c_values = [scenario_data[scenario_data['year'] == y]['C端收入'].values[0] for y in years]
        
        ax.bar(x_pos + i * bar_width, b2b_values, bar_width, label=f'{scenario}-B端', color=colors[0], alpha=0.7)
        ax.bar(x_pos + i * bar_width, c_values, bar_width, bottom=b2b_values, label=f'{scenario}-C端', color=colors[1], alpha=0.7)
    
    ax.set_xlabel('年份', fontproperties="SimHei")
    ax.set_ylabel('收入 ($)', fontproperties="SimHei")
    ax.set_title('三阶段推广路径的收入构成分析', fontproperties="SimHei")
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(years)
    ax.legend(prop={"family": "SimHei"})
    plt.tight_layout()
    plt.savefig("revenue_composition.png", dpi=300, bbox_inches='tight')
    plt.show()
# ==== 主流程 ====
if __name__ == "__main__":
    print("正在生成敏感性分析热力图...")
    sensitivity_heatmap(params)

    print("正在生成K线图...")
    profit_candlestick(params)

    print("正在生成三阶段财务对比图...")
    summary_charts(params)
