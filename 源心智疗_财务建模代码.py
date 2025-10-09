# -*- coding: utf-8 -*-
# 源心智疗 财务建模与蒙特卡洛仿真 + 热力图 + K线图
# ==============================================
# 本脚本主要功能：
# 1️⃣ 定义基础业务参数与三种情景（保守 / 基础 / 激进）；
# 2️⃣ 计算各情景下未来3年净现值（NPV）；
# 3️⃣ 进行敏感性分析（采用率 × 单价）热力图；
# 4️⃣ 基于Monte Carlo仿真生成年度利润“K线图”；
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 参数定义 ====
params = {
    "pilot_universities": 5,           # 参与试点的高校数量
    "students_per_uni": 20000,         # 每所高校的学生数
    "scenarios": {                     # 不同市场情景假设
        "conservative": {              # 保守场景
            "adoption_rate": 0.05,     # 采用率（5%的学生使用）
            "price_per_user_year": 1.0,# 每用户每年收入（美元）
            "hosting_cost_per_user_month": 1.0,  # 每用户每月服务器成本
            "other_opex_pct": 0.35     # 其他运营成本占收入比例
        },
        "base": {                      # 基础场景
            "adoption_rate": 0.10,
            "price_per_user_year": 10.0,
            "hosting_cost_per_user_month": 0.5,
            "other_opex_pct": 0.20
        },
        "aggressive": {                # 激进场景
            "adoption_rate": 0.20,
            "price_per_user_year": 30.0,
            "hosting_cost_per_user_month": 0.3,
            "other_opex_pct": 0.15
        },
    },
    "mvp_dev_cost_usd": 40000.0,       # MVP产品开发成本
    "cac_per_uni": 5000.0,             # 每高校获客成本（CAC）
    "gpu_monthly_cost_usd": 500.0,     # GPU服务器每月成本
    "ops_salary_annual_usd": 30000.0,  # 运维年薪
    "discount_rate": 0.10,             # 折现率（10%）
    "horizon_years": 3,                # 模型周期（3年）
    "growth_rate_users": 0.25          # 用户年增长率（25%）
}


# ==== 场景财务计算 ====
def scenario_financials(params, scen_key):
    """
    输入：参数字典 + 场景键（conservative / base / aggressive）
    输出：该场景下的年度现金流表和3年净现值（NPV）
    """
    s = params["scenarios"][scen_key]

    # 计算总用户与收入
    total_students = params["pilot_universities"] * params["students_per_uni"]  # 总学生数
    active_users = total_students * s["adoption_rate"]                           # 活跃用户数
    annual_revenue = active_users * s["price_per_user_year"]                     # 年收入

    # 年度成本项
    hosting_annual = active_users * s["hosting_cost_per_user_month"] * 12.0      # 服务器成本
    gpu_annual = params["gpu_monthly_cost_usd"] * 12.0                           # GPU成本
    ops_annual = params["ops_salary_annual_usd"]                                 # 运维成本
    cac_total = params["cac_per_uni"] * params["pilot_universities"]             # 获客成本总额
    other_opex = s["other_opex_pct"] * annual_revenue                            # 其他运营费用

    # EBITDA与净利润
    annual_costs = hosting_annual + gpu_annual + ops_annual + other_opex
    ebitda = annual_revenue - annual_costs
    dev_amort_annual = params["mvp_dev_cost_usd"] / params["horizon_years"]      # 开发成本摊销
    net_profit = ebitda - dev_amort_annual - cac_total / params["horizon_years"] # 净利润

    # 构建未来3年现金流表
    cashflows = []
    users_year0 = active_users
    for y in range(1, params["horizon_years"] + 1):
        users_y = users_year0 * ((1 + params["growth_rate_users"]) ** (y - 1))   # 用户增长
        rev_y = users_y * s["price_per_user_year"]                               # 收入
        host_y = users_y * s["hosting_cost_per_user_month"] * 12.0               # 托管成本
        other_y = s["other_opex_pct"] * rev_y                                    # 其他运营成本
        costs_y = host_y + params["gpu_monthly_cost_usd"] * 12.0 + params["ops_salary_annual_usd"] + other_y
        ebitda_y = rev_y - costs_y                                               # EBITDA
        dev_amort_y = params["mvp_dev_cost_usd"] / params["horizon_years"]
        cac_amort_y = params["cac_per_uni"] * params["pilot_universities"] / params["horizon_years"]
        net_y = ebitda_y - dev_amort_y - cac_amort_y                             # 净利润
        cashflows.append({
            "year": y,
            "users": round(users_y),
            "revenue": round(rev_y, 2),
            "costs": round(costs_y, 2),
            "ebitda": round(ebitda_y, 2),
            "net_profit": round(net_y, 2)
        })

    # 计算NPV（净现值）
    dr = params["discount_rate"]
    npv = sum([cf["net_profit"] / ((1 + dr) ** cf["year"]) for cf in cashflows])
    return {"scenario": scen_key, "cashflows": cashflows, "npv_3yr": round(npv, 2)}


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
            tmp_scen = params["scenarios"]["base"].copy()
            tmp_scen["price_per_user_year"] = p
            tmp_scen["adoption_rate"] = a
            tmp["scenarios"] = {"sim": tmp_scen}
            res = scenario_financials(tmp, "sim")
            npv_matrix[i, j] = res["npv_3yr"]

    # 绘制热力图
    plt.figure(figsize=(7, 6))
    sns.heatmap(npv_matrix, xticklabels=np.round(adoption_rates, 2), yticklabels=np.round(prices, 2),
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
        p = params["scenarios"]["base"]["price_per_user_year"] * np.random.uniform(0.7, 1.3)
        a = params["scenarios"]["base"]["adoption_rate"] * np.random.uniform(0.5, 1.5)
        h = params["scenarios"]["base"]["hosting_cost_per_user_month"] * np.random.uniform(0.8, 1.2)
        tmp = params.copy()
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

# ==== 寻常图表（收入、利润、NPV对比） ====
def summary_charts(params):
    scenarios = ["conservative", "base", "aggressive"]
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

    # 1️⃣ 年度收入对比柱状图
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="year", y="revenue", hue="scenario", palette="Set2")
    plt.title("不同情景下的年度收入对比", fontproperties="SimHei")
    plt.xlabel("年份", fontproperties="SimHei")
    plt.ylabel("年度收入 ($)", fontproperties="SimHei")
    plt.legend(prop={"family": "SimHei"})
    plt.tight_layout()
    plt.savefig("annual_revenue_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2️⃣ 年度净利润折线图
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=df, x="year", y="net_profit", hue="scenario", marker="o")
    plt.title("不同情景下的年度净利润变化", fontproperties="SimHei")
    plt.xlabel("年份", fontproperties="SimHei")
    plt.ylabel("净利润 ($)", fontproperties="SimHei")
    plt.legend(prop={"family": "SimHei"})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("net_profit_trend.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3️⃣ NPV 对比条形图
    npv_df = df.groupby("scenario")["npv"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=npv_df, x="scenario", y="npv", palette="viridis")
    plt.title("三种情景的净现值（NPV）对比", fontproperties="SimHei")
    plt.xlabel("情景", fontproperties="SimHei")
    plt.ylabel("三年净现值 ($)", fontproperties="SimHei")
    plt.tight_layout()
    plt.savefig("npv_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==== 主流程 ====
if __name__ == "__main__":
    print("正在生成敏感性分析热力图...")
    sensitivity_heatmap(params)

    print("正在生成K线图...")
    profit_candlestick(params)

    print("正在生成基础财务对比图...")
    summary_charts(params)
