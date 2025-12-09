"""
反向求解器测试文件（优化版）
- 全局加载数据（只加载一次）
- 每个测试使用不同的随机种子
- 完整的α敏感性分析（0.05-0.95）
"""

import sys
import os
import pickle
import gzip
from datetime import datetime
import numpy as np
import time as time_module
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from reverse_solver_pseudocode import ReverseLabelSettingSolver
import config as config
from visualization_generator import generate_html_with_svg
import time
# ═══════════════════════════════════════════════════════════════════
# 全局变量：数据（只加载一次）
# ═══════════════════════════════════════════════════════════════════

# 全局数据变量
G_GLOBAL = None
SPARSE_DATA_GLOBAL = None
NODE_TO_INDEX_GLOBAL = None
SCENARIO_DATES_GLOBAL = None
SCENARIO_PROBS_GLOBAL = None
TIME_INTERVALS_PER_DAY_GLOBAL = None

DATA_LOADED = False


def load_data_once(data_path=None):
    """全局加载数据（只加载一次）"""
    global G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL
    global SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL
    global DATA_LOADED
    
    if DATA_LOADED:
        return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
                SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)
    
    if data_path is None:
        data_path = config.DATA_PATH
    
    print(f"\n{'='*70}")
    print(f"加载测试数据（仅加载一次）")
    print(f"{'='*70}")
    print(f"  数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    start_time = time_module.time()
    
    with gzip.open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    G_GLOBAL = data['G']
    SPARSE_DATA_GLOBAL = data['sparse_data']
    NODE_TO_INDEX_GLOBAL = data['node_to_index']
    SCENARIO_DATES_GLOBAL = [datetime.strptime(d, '%Y-%m-%d').date() 
                             for d in data['scenario_dates']]
    SCENARIO_PROBS_GLOBAL = data['scenario_probs']
    TIME_INTERVALS_PER_DAY_GLOBAL = data['time_intervals_per_day']
    
    load_time = time_module.time() - start_time
    
    print(f"  ✓ 加载成功 (用时 {load_time:.2f}秒)")
    print(f"  节点数: {len(G_GLOBAL.nodes()):,}")
    print(f"  边数: {len(G_GLOBAL.edges()):,}")
    print(f"  场景数: {len(SCENARIO_DATES_GLOBAL)}")
    print(f"  时间片数/天: {TIME_INTERVALS_PER_DAY_GLOBAL:,}")
    print(f"{'='*70}\n")
    
    DATA_LOADED = True
    
    return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
            SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)


def get_data():
    """获取全局数据"""
    if not DATA_LOADED:
        return load_data_once()
    return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
            SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def select_od_pair(node_to_index):
    """选择OD对（使用指定种子）"""
    nodes = list(node_to_index.keys())
    np.random.seed(int(time.time()))
    origin = np.random.choice(nodes)
    destination = np.random.choice([n for n in nodes if n != origin])
    return origin, destination


def time_to_string(time_01min):
    """将0.1分钟单位转换为HH:MM格式"""
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"



# ═══════════════════════════════════════════════════════════════════
# 测试1: 绘图
# ═══════════════════════════════════════════════════════════════════

def plot_all_paths_distributions(result, alpha, target_arrival_time, 
                                 output_file='all_paths_distributions.png'):
    """
    绘制所有候选路径的出发时间分布对比图
    
    Args:
        result: solve()的返回结果（包含all_paths）
        alpha: 可靠性参数
        target_arrival_time: 目标到达时间
        output_file: 输出文件路径
    """
    
    if 'all_paths' not in result or not result['all_paths']:
        print("没有保存的路径分布数据")
        return
    
    all_paths = result['all_paths']
    
    print(f"\n{'='*70}")
    print(f"绘制所有候选路径的出发时间分布")
    print(f"{'='*70}")
    print(f"  候选路径数量: {len(all_paths)}")
    print(f"  可靠性α: {alpha}")
    print(f"  目标到达时间: {time_to_string(target_arrival_time)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 创建图表
    # ═══════════════════════════════════════════════════════════════════
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图1: 所有分布的CDF对比
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax1 = axes[0, 0]
    
    best_path_idx = None
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_paths)))
    
    for i, path_info in enumerate(all_paths):
        dist = path_info['distribution']
        values = sorted(dist.values)
        cdf = np.arange(1, len(values) + 1) / len(values)
        
        if path_info['is_best']:
            # 最优路径用红色粗线
            ax1.plot([v/10 for v in values], cdf, 
                    color='red', linewidth=3, alpha=0.9,
                    label=f'最优路径 (路径{i+1})', zorder=10)
            best_path_idx = i
        else:
            # 其他路径用半透明细线
            ax1.plot([v/10 for v in values], cdf, 
                    color=colors[i], linewidth=1, alpha=0.3)
    
    # 标记关键分位数
    ax1.axhline(1 - alpha, color='orange', linestyle='--', linewidth=2,
                label=f'α={alpha} 分位数线')
    ax1.axvline(target_arrival_time/10, color='green', linestyle='--', linewidth=2,
                label='目标到达时间')
    
    ax1.set_xlabel('出发时间 (分钟)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('累积概率 (CDF)', fontsize=12, fontweight='bold')
    ax1.set_title(f'所有候选路径的CDF对比 (共{len(all_paths)}条)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图2: 关键分位数对比（箱线图风格）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax2 = axes[0, 1]
    
    quantiles_to_show = [0.05, 0.25, 0.5, 0.75, 0.95, 1-alpha]
    quantile_labels = ['Q5', 'Q25', 'Q50', 'Q75', 'Q95', f'Q{int((1-alpha)*100)}']
    
    x_positions = np.arange(len(all_paths))
    width = 0.12
    
    for q_idx, (q, label) in enumerate(zip(quantiles_to_show, quantile_labels)):
        q_values = []
        for path_info in all_paths:
            q_val = path_info['distribution'].get_quantile(q) / 10
            q_values.append(q_val)
        
        offset = (q_idx - len(quantiles_to_show)/2) * width
        ax2.bar(x_positions + offset, q_values, width, 
               label=label, alpha=0.7)
    
    # 高亮最优路径
    if best_path_idx is not None:
        ax2.axvline(best_path_idx, color='red', linestyle='--', 
                   linewidth=2, alpha=0.5, label='最优路径')
    
    ax2.set_xlabel('路径编号', fontsize=12, fontweight='bold')
    ax2.set_ylabel('出发时间 (分钟)', fontsize=12, fontweight='bold')
    ax2.set_title('各路径的关键分位数对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'P{i+1}' for i in range(len(all_paths))], rotation=45)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图3: 最优路径 vs 次优路径的详细对比
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax3 = axes[1, 0]
    
    if best_path_idx is not None and len(all_paths) > 1:
        # 找到次优路径（latest_departure第二大的）
        sorted_paths = sorted(enumerate(all_paths), 
                            key=lambda x: x[1]['latest_departure'], 
                            reverse=True)
        
        best_idx, best_info = sorted_paths[0]
        second_idx, second_info = sorted_paths[1] if len(sorted_paths) > 1 else (None, None)
        
        # 绘制最优路径分布
        best_values = sorted(best_info['distribution'].values)
        best_cdf = np.arange(1, len(best_values) + 1) / len(best_values)
        ax3.plot([v/10 for v in best_values], best_cdf,
                'r-', linewidth=3, label=f'最优路径 (P{best_idx+1})')
        
        # 绘制次优路径分布
        if second_info:
            second_values = sorted(second_info['distribution'].values)
            second_cdf = np.arange(1, len(second_values) + 1) / len(second_values)
            ax3.plot([v/10 for v in second_values], second_cdf,
                    'b--', linewidth=2, label=f'次优路径 (P{second_idx+1})')
        
        # 标记α分位数
        best_q = best_info['distribution'].get_quantile(1-alpha) / 10
        ax3.axvline(best_q, color='red', linestyle=':', linewidth=2,
                   label=f'最优路径 α-分位数')
        
        if second_info:
            second_q = second_info['distribution'].get_quantile(1-alpha) / 10
            ax3.axvline(second_q, color='blue', linestyle=':', linewidth=2,
                       label=f'次优路径 α-分位数')
        
        ax3.axhline(1-alpha, color='orange', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('出发时间 (分钟)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('累积概率', fontsize=12, fontweight='bold')
        ax3.set_title('最优 vs 次优路径详细对比', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(alpha=0.3)
        
        # 添加文本说明
        info_text = f"最优路径:\n"
        info_text += f"  最晚出发: {time_to_string(best_info['latest_departure'])}\n"
        info_text += f"  期望出发: {time_to_string(best_info['expected_departure'])}\n"
        info_text += f"  路径长度: {len(best_info['path'])}\n"
        if second_info:
            info_text += f"\n次优路径:\n"
            info_text += f"  最晚出发: {time_to_string(second_info['latest_departure'])}\n"
            info_text += f"  期望出发: {time_to_string(second_info['expected_departure'])}\n"
            info_text += f"  路径长度: {len(second_info['path'])}\n"
        
        ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图4: 统计信息汇总
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 创建统计表格
    stats_data = []
    for i, path_info in enumerate(all_paths):
        stats_data.append([
            f"路径{i+1}" + ('⭐' if path_info['is_best'] else ''),
            f"{len(path_info['path'])}",
            time_to_string(path_info['latest_departure']),
            time_to_string(path_info['expected_departure']),
            f"{path_info['std_departure']/10:.1f}",
            f"{(target_arrival_time - path_info['latest_departure'])/10:.1f}"
        ])
    
    table = ax4.table(
        cellText=stats_data,
        colLabels=['路径', '长度', '最晚出发', '期望出发', '标准差(分)', '预留(分)'],
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.10, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 高亮最优路径
    if best_path_idx is not None:
        for i in range(6):
            table[(best_path_idx + 1, i)].set_facecolor('#ffcccc')
    
    ax4.set_title('候选路径统计汇总', fontsize=14, fontweight='bold', pad=20)
    
    # ═══════════════════════════════════════════════════════════════════
    # 保存图表
    # ═══════════════════════════════════════════════════════════════════
    
    plt.suptitle(f'α={alpha} 时所有候选路径的出发时间分布对比', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ 图表已保存: {output_file}\n")
    plt.close()


def time_to_string(time_01min):
    """时间格式转换"""
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"




# ═══════════════════════════════════════════════════════════════════
# 测试1: 基本求解
# ═══════════════════════════════════════════════════════════════════

def test_1_basic_solve():
    """测试1: 基本求解"""
    print(f"\n{'='*70}")
    print(f"测试1: 基本求解")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 初始化求解器
    mode = config.get_mode_config('standard')
    
    solver = ReverseLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=mode['L1'],
        L2=mode['L2'],
        verbose=config.REVERSE_VERBOSE
    )
    
    # 选择OD对（使用测试1的种子）
    origin, destination = select_od_pair(node_to_index)
    print(f"  测试OD对 (seed=1001): {origin} → {destination}")
    
    # 设置问题参数
    target_arrival_time = (config.DEFAULT_ARRIVAL_HOUR * 60 + 
                          config.DEFAULT_ARRIVAL_MINUTE) * 10
    alpha = config.REVERSE_ALPHA_DEFAULT
    
    print(f"  目标到达时间: {time_to_string(target_arrival_time)}")
    print(f"  可靠性要求: α={alpha}\n")
    
    # 求解
    result = solver.solve(
        origin=origin,
        destination=destination,
        target_arrival_time=target_arrival_time,
        alpha=alpha,
        max_labels=mode['max_labels']
    )
    
    # 验证结果
    print(f"\n{'─'*70}")
    print(f"测试1验证")
    print(f"{'─'*70}")
    
    assert result['success'], "❌ 求解失败"
    print(f"  ✓ 求解成功")
    
    assert result['path'] is not None, "❌ 路径为空"
    print(f"  ✓ 路径非空 (长度: {len(result['path'])})")
    
    assert result['path'][0] == origin, "❌ 起点不匹配"
    print(f"  ✓ 起点正确: {origin}")
    
    assert result['path'][-1] == destination, "❌ 终点不匹配"
    print(f"  ✓ 终点正确: {destination}")
    
    assert result['latest_departure_time'] > 0, "❌ 最晚出发时间无效"
    print(f"  ✓ 最晚出发时间: {time_to_string(result['latest_departure_time'])}")
    
    assert result['reserved_time'] > 0, "❌ 预留时间无效"
    print(f"  ✓ 预留时间: {result['reserved_time']/10:.1f}分钟")
    
    assert result['latest_departure_time'] < target_arrival_time, "❌ 出发晚于到达"
    print(f"  ✓ 时间逻辑正确")
    
    print(f"\n  🎉 测试1通过！")
    print(f"{'='*70}\n")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# 测试2: α敏感性分析（完整版：0.05-0.95）
# ═══════════════════════════════════════════════════════════════════

def test_2_alpha_sensitivity():
    """测试2: α敏感性分析（0.05-0.95）"""
    print(f"\n{'='*70}")
    print(f"测试2: α敏感性分析（完整版）")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    mode = config.get_mode_config('fast')
    
    solver = ReverseLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=mode['L1'],
        L2=mode['L2'],
        verbose=False
    )
    
    # 使用测试2的种子
    origin, destination = select_od_pair(node_to_index)
    target_arrival_time = 9 * 60 * 10  # 09:00
    
    print(f"  测试OD对 (seed=2002): {origin} → {destination}")
    print(f"  目标到达时间: {time_to_string(target_arrival_time)}\n")
    
    # 完整的α值范围：0.05, 0.10, 0.15, .. ., 0.95
    alphas = np.arange(0.05, 1.00, 0.05).round(2).tolist()
    
    print(f"  测试α值范围: 0.05 到 0. 95 (步长0.05)")
    print(f"  总共 {len(alphas)} 个测试点\n")
    
    results = []
    # detailed_alphas = [0.50, 0.75, 0.95]  # 中、高、很高可靠性
    detailed_alphas = alphas
    detailed_results = {}
    
    print(f"  开始测试:")
    for i, alpha in enumerate(alphas, 1):
        print(f"    [{i:2d}/{len(alphas)}] α={alpha:.2f}.. .", end='', flush=True)
        
        save_all = alpha in detailed_alphas

        result = solver.solve(
            origin, destination, target_arrival_time, alpha,
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            result_data = {
                'alpha': alpha,
                'latest_departure': result['latest_departure_time'],
                'expected_departure': result['expected_departure_time'],
                'reserved_time': result['reserved_time'],
                'path': result['path'],
                'path_length': len(result['path']),
                'target_arrival': target_arrival_time,
                'distribution': result['distribution']  # ← 保存分布用于可视化
            }
            
            # ✅ 如果保存了所有路径，添加到详细结果
            if save_all and 'all_paths' in result:
                result_data['all_paths'] = result['all_paths']
                result_data['num_candidates'] = result['num_candidate_paths']
                detailed_results[alpha] = result_data
                print(f" ✓ 最晚={time_to_string(result['latest_departure_time'])}, "
                      f"预留={result['reserved_time']/10:.1f}分, "
                      f"候选路径={result['num_candidate_paths']}")
            else:
                print(f" ✓ 最晚={time_to_string(result['latest_departure_time'])}, "
                      f"预留={result['reserved_time']/10:.1f}分")
            
            results.append(result_data)

                # 绘制所有候选路径的分布对比
            # plot_all_paths_distributions(
            #     result, 
            #     analysis_alpha, 
            #     target_arrival_time,
            #     output_file=f'result/alpha_{int(analysis_alpha*100)}_all_paths.png'
            # )
        else:
            print(f" ✗ 失败")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 验证
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'─'*70}")
    print(f"测试2验证")
    print(f"{'─'*70}")
    
    success_rate = len(results) / len(alphas) * 100
    print(f"  成功率: {len(results)}/{len(alphas)} ({success_rate:.1f}%)")
    
    assert len(results) >= len(alphas) * 0.7, \
        f"❌ 成功求解的α值太少: {len(results)}/{len(alphas)}"
    print(f"  ✓ 成功率达标 (≥70%)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 打印详细对比表（全部使用HH:MM格式）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n  α敏感性详细对比:")
    print(f"  {'α值':<8} {'最晚出发':<12} {'期望出发':<12} {'目标到达':<12} "
          f"{'预留(分)':<12} {'路径长度':<10}")
    print(f"  {'-'*80}")
    
    # 显示所有结果（或部分关键点）
    display_all = len(results) <= 10
    
    if display_all:
        for r in results:
            print(f"  {r['alpha']:<8.2f} "
                  f"{time_to_string(r['latest_departure']):<12} "
                  f"{time_to_string(r['expected_departure']):<12} "
                  f"{time_to_string(r['target_arrival']):<12} "
                  f"{r['reserved_time']/10:<12.1f} "
                  f"{r['path_length']:<10}")
    else:
        # 显示关键点
        key_indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]
        for i in key_indices:
            if i < len(results):
                r = results[i]
                print(f"  {r['alpha']:<8.2f} "
                      f"{time_to_string(r['latest_departure']):<12} "
                      f"{time_to_string(r['expected_departure']):<12} "
                      f"{time_to_string(r['target_arrival']):<12} "
                      f"{r['reserved_time']/10:<12.1f} "
                      f"{r['path_length']:<10}")
        print(f"  ...  (显示 {len(key_indices)}/{len(results)} 个结果，完整结果见输出文件)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 打印路径详情（选择几个代表性的α值）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n  路径详情（代表性α值）:")
    print(f"  {'-'*70}")
    
    # 选择3个代表性α值：低、中、高
    representative_indices = []
    if len(results) > 0:
        representative_indices.append(0)
    if len(results) >= 2:
        representative_indices.append(len(results)//2)
    if len(results) >= 3:
        representative_indices.append(-1)
    
    for idx in representative_indices:
        if idx < len(results):
            r = results[idx]
            print(f"\n  【α = {r['alpha']:.2f}】")
            print(f"    起点: {origin}")
            print(f"    终点: {destination}")
            print(f"    路径: {format_path(r['path'])}")
            print(f"    路径长度: {r['path_length']} 个节点")
            print(f"    ┌─ 时间信息 ─────────────────────────────────────────┐")
            print(f"    │ 最晚出发时间: {time_to_string(r['latest_departure']):<10} "
                  f"({format_minutes(r['latest_departure'])})  │")
            print(f"    │ 期望出发时间: {time_to_string(r['expected_departure']):<10} "
                  f"({format_minutes(r['expected_departure'])})  │")
            print(f"    │ 目标到达时间: {time_to_string(r['target_arrival']):<10} "
                  f"({format_minutes(r['target_arrival'])})  │")
            print(f"    │ 预留时间:     {r['reserved_time']/10:>6.1f} 分钟"
                  f"{' '*26}│")
            print(f"    └────────────────────────────────────────────────────┘")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 验证单调性
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n  单调性检查 (抽样验证):")
    monotonic_violations = 0
    check_indices = [i for i in range(len(results)-1) if i % 3 == 0]
    
    for i in check_indices:
        curr = results[i]
        next_r = results[i+1]
        
        # α增大时，最晚出发时间应该减小或相近（容差10分钟）
        if curr['latest_departure'] < next_r['latest_departure'] - 100:
            monotonic_violations += 1
            print(f"    ⚠ α={curr['alpha']:.2f}→{next_r['alpha']:.2f}: "
                  f"最晚出发反而增加 "
                  f"({time_to_string(curr['latest_departure'])} → "
                  f"{time_to_string(next_r['latest_departure'])})")
    
    if monotonic_violations == 0:
        print(f"    ✓ 所有检查点符合单调性")
    else:
        print(f"    ⚠ {monotonic_violations}/{len(check_indices)} 个点违反单调性 "
              f"({monotonic_violations/len(check_indices)*100:.1f}%)")
    
    try:
        save_alpha_sensitivity_results(results, origin, destination, target_arrival_time)
        print(f"\n  ✓ 详细结果已保存到: alpha_sensitivity_results.txt")
    except Exception as e:
        print(f"\n  ⚠ 保存结果失败: {e}")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 可视化
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if config.SHOW_PLOTS and len(results) >= 5:
        try:
            plot_alpha_sensitivity(results, target_arrival_time)
        except Exception as e:
            print(f"    ⚠ 可视化失败: {e}")
    
    print(f"\n  🎉 测试2通过！")
    print(f"{'='*70}\n")
    
    return {
        'all_results': results,
        'detailed_results': detailed_results,  # ← 新增：包含所有候选路径的详细结果
        'origin': origin,
        'destination': destination,
        'target_arrival_time': target_arrival_time
    }

def save_alpha_sensitivity_results(results, origin, destination, target_arrival_time):
    """保存α敏感性分析详细结果到文件（全部使用HH:MM格式）"""
    filename = 'result/alpha_sensitivity_results.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("α敏感性分析详细结果\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"起点: {origin}\n")
        f.write(f"终点: {destination}\n")
        f.write(f"目标到达时间: {time_to_string(target_arrival_time)} "
                f"({format_minutes(target_arrival_time)})\n")
        f.write(f"测试α值数量: {len(results)}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "="*90 + "\n\n")
        
        # 汇总表格
        f.write("【汇总表格】\n")
        f.write("-"*90 + "\n")
        f.write(f"{'α值':<8} {'最晚出发':<12} {'期望出发':<12} {'目标到达':<12} "
                f"{'预留(分)':<12} {'路径长度':<10}\n")
        f.write("-"*90 + "\n")
        
        for r in results:
            f.write(f"{r['alpha']:<8.2f} "
                    f"{time_to_string(r['latest_departure']):<12} "
                    f"{time_to_string(r['expected_departure']):<12} "
                    f"{time_to_string(r['target_arrival']):<12} "
                    f"{r['reserved_time']/10:<12.1f} "
                    f"{r['path_length']:<10}\n")
        
        # 详细路径信息
        f.write("\n" + "="*90 + "\n")
        f.write("【详细路径信息】\n")
        f.write("="*90 + "\n")
        
        for r in results:
            f.write(f"\n{'─'*90}\n")
            f.write(f"α = {r['alpha']:.2f}\n")
            f.write(f"{'─'*90}\n")
            
            f.write(f"路径摘要: {format_path(r['path'])}\n")
            f.write(f"完整路径: {' → '.join(map(str, r['path']))}\n")
            f.write(f"路径长度: {r['path_length']} 个节点\n\n")
            
            f.write(f"时间信息:\n")
            f.write(f"  最晚出发时间: {time_to_string(r['latest_departure'])} "
                    f"({format_minutes(r['latest_departure'])})\n")
            f.write(f"  期望出发时间: {time_to_string(r['expected_departure'])} "
                    f"({format_minutes(r['expected_departure'])})\n")
            f.write(f"  目标到达时间: {time_to_string(r['target_arrival'])} "
                    f"({format_minutes(r['target_arrival'])})\n")
            f.write(f"  预留时间:     {r['reserved_time']/10:.1f} 分钟\n")  # 修正这里
            
            # 时间差异分析
            time_diff = r['expected_departure'] - r['latest_departure']
            f.write(f"  出发时间差异: {time_diff/10:.1f} 分钟 "
                    f"(期望 - 最晚)\n")
        
        # 统计信息
        f.write("\n" + "="*90 + "\n")
        f.write("【统计信息】\n")
        f.write("="*90 + "\n\n")
        
        reserved_times = [r['reserved_time']/10 for r in results]
        path_lengths = [r['path_length'] for r in results]
        
        f.write(f"预留时间统计:\n")
        f.write(f"  最小值: {min(reserved_times):.1f} 分钟 (α={results[np.argmin(reserved_times)]['alpha']:.2f})\n")
        f.write(f"  最大值: {max(reserved_times):.1f} 分钟 (α={results[np.argmax(reserved_times)]['alpha']:.2f})\n")
        f.write(f"  平均值: {np.mean(reserved_times):.1f} 分钟\n")
        f.write(f"  标准差: {np.std(reserved_times):.1f} 分钟\n\n")  # 修正这里
        
        f.write(f"路径长度统计:\n")
        f.write(f"  最小值: {min(path_lengths)} 个节点\n")
        f.write(f"  最大值: {max(path_lengths)} 个节点\n")
        f.write(f"  平均值: {np.mean(path_lengths):.1f} 个节点\n")
        
        # 单调性分析
        f.write(f"\n单调性分析:\n")
        violations = 0
        for i in range(len(results)-1):
            if results[i]['latest_departure'] < results[i+1]['latest_departure'] - 100:
                violations += 1
        
        f.write(f"  检查点数: {len(results)-1}\n")
        f.write(f"  违反单调性: {violations} 个\n")
        f.write(f"  单调性率: {(1-violations/(len(results)-1))*100:.1f}%\n")

def time_to_string(time_01min):
    """
    将0. 1分钟单位转换为HH:MM格式
    
    Args:
        time_01min: 时间（0.1分钟单位）
        
    Returns:
        HH:MM格式的字符串
    """
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def format_minutes(time_01min):
    """
    格式化分钟数（带单位）
    
    Args:
        time_01min: 时间（0.1分钟单位）
        
    Returns:
        格式化的字符串，如 "505. 0分钟"
    """
    minutes = time_01min / 10
    return f"{minutes:.1f}分钟"


def format_path(path):
    """格式化路径输出"""
    if len(path) <= 10:
        return ' → '.join(map(str, path))
    else:
        return (f"{' → '.join(map(str, path[:5]))} → ...  "
                f"→ {' → '.join(map(str, path[-3:]))}")

# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def time_to_string(time_01min):
    """
    将0.1分钟单位转换为HH:MM格式
    
    Args:
        time_01min: 时间（0.1分钟单位）
        
    Returns:
        HH:MM格式的字符串
    """
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def plot_alpha_sensitivity(results, target_arrival_time):
    """绘制α敏感性分析图"""
    alphas = [r['alpha'] for r in results]
    latest_deps = [r['latest_departure']/10 for r in results]
    reserved_times = [r['reserved_time']/10 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: 最晚出发时间
    ax1.plot(alphas, latest_deps, 'b-o', linewidth=2, markersize=4, label='Latest Departure')
    ax1.axhline(target_arrival_time/10, color='orange', linestyle='--', 
                linewidth=2, label='Target Arrival')
    ax1.set_xlabel('Reliability α', fontsize=12)
    ax1.set_ylabel('Departure Time (minutes)', fontsize=12)
    ax1.set_title('α Sensitivity - Departure Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 子图2: 预留时间
    ax2.plot(alphas, reserved_times, 'r-s', linewidth=2, markersize=4, label='Reserved Time')
    ax2.set_xlabel('Reliability α', fontsize=12)
    ax2.set_ylabel('Reserved Time (minutes)', fontsize=12)
    ax2.set_title('α Sensitivity - Reserved Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result/alpha_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ 可视化已保存: alpha_sensitivity_analysis.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 测试3: 性能测试
# ═══════════════════════════════════════════════════════════════════

def test_3_performance():
    """测试3: 性能测试"""
    print(f"\n{'='*70}")
    print(f"测试3: 性能测试")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 使用测试3的种子
    origin, destination = select_od_pair(node_to_index)
    target_arrival_time = 9 * 60 * 10
    
    print(f"  测试OD对 (seed=3003): {origin} → {destination}")
    print(f"  目标到达: {time_to_string(target_arrival_time)}\n")
    
    # 测试不同配置
    test_configs = [
        ('快速模式', config.FAST_MODE),
        ('标准模式', config.STANDARD_MODE),
    ]
    
    performance_results = []
    
    for config_name, mode in test_configs:
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  配置: {config_name}")
        print(f"    L1={mode['L1']}, L2={mode['L2']}, 最大标签={mode['max_labels']:,}")
        
        solver = ReverseLabelSettingSolver(
            G, sparse_data, node_to_index, scenario_dates,
            scenario_probs, time_intervals_per_day,
            L1=mode['L1'],
            L2=mode['L2'],
            verbose=False
        )
        
        start = time_module.time()
        result = solver.solve(
            origin, destination, target_arrival_time, 0.95,
            max_labels=mode['max_labels']
        )
        elapsed = time_module.time() - start
        
        if result['success']:
            perf_data = {
                'config': config_name,
                'L1': mode['L1'],
                'L2': mode['L2'],
                'time': elapsed,
                'iterations': result['iterations'],
                'labels_generated': result['stats']['labels_generated'],
                'labels_dominated': result['stats']['labels_dominated'],
                'pruning_rate': result['stats']['labels_dominated']/result['stats']['labels_generated']*100
            }
            performance_results.append(perf_data)
            
            print(f"    ✓ 成功")
            print(f"      耗时: {elapsed:.2f}秒")
            print(f"      迭代: {result['iterations']}")
            print(f"      生成标签: {result['stats']['labels_generated']:,}")
            print(f"      剪枝率: {perf_data['pruning_rate']:.1f}%")
            print(f"      最晚出发: {time_to_string(result['latest_departure_time'])}")
        else:
            print(f"    ✗ 失败")
    
    # 性能对比
    if len(performance_results) >= 2:
        print(f"\n  性能对比:")
        fast = performance_results[0]
        standard = performance_results[1]
        speedup = standard['time'] / fast['time']
        print(f"    快速模式 vs 标准模式:")
        print(f"      速度提升: {speedup:.2f}x")
        print(f"      标签数对比: {fast['labels_generated']:,} vs {standard['labels_generated']:,}")
    
    print(f"\n  🎉 测试3完成！")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 测试4: 时间一致性
# ═══════════════════════════════════════════════════════════════════

def test_4_time_consistency():
    """测试4: 时间一致性"""
    print(f"\n{'='*70}")
    print(f"测试4: 时间一致性检查")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    mode = config.get_mode_config('fast')
    
    solver = ReverseLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=mode['L1'],
        L2=mode['L2'],
        verbose=False
    )
    
    # 使用测试4的种子
    origin, destination = select_od_pair(node_to_index)
    
    print(f"  测试OD对 (seed=4004): {origin} → {destination}\n")
    
    # 测试不同到达时间
    test_times = config.TIME_BUDGET_TEST_TIMES[:3]
    results = []
    
    for hour, minute in test_times:
        target_time = (hour * 60 + minute) * 10
        time_str = f"{hour:02d}:{minute:02d}"
        
        print(f"  测试目标到达: {time_str}...", end='', flush=True)
        
        result = solver.solve(
            origin, destination, target_time, 0.95,
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            results.append({
                'target_time': target_time,
                'time_str': time_str,
                'latest_dep': result['latest_departure_time'],
                'reserved': result['reserved_time']
            })
            print(f" ✓ 最晚出发={time_to_string(result['latest_departure_time'])}")
        else:
            print(f" ✗ 失败")
    
    # 验证
    print(f"\n{'─'*70}")
    print(f"测试4验证")
    print(f"{'─'*70}")
    
    for r in results:
        # 验证时间逻辑
        assert r['latest_dep'] < r['target_time'], \
            f"❌ 时间逻辑错误: {r['time_str']}"
        print(f"  ✓ {r['time_str']}: 时间逻辑正确")
        
        # 验证预留时间
        expected_reserved = r['target_time'] - r['latest_dep']
        assert abs(r['reserved'] - expected_reserved) < 1, \
            f"❌ 预留时间计算错误: {r['time_str']}"
        print(f"    预留时间: {r['reserved']/10:.1f}分")
    
    print(f"\n  🎉 测试4通过！")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 测试5: 多OD对测试
# ═══════════════════════════════════════════════════════════════════

def test_5_multiple_od_pairs():
    """测试5: 多OD对测试（测试算法稳定性）- 修改版"""
    print(f"\n{'='*70}")
    print(f"测试5: 多OD对测试")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    mode = config.get_mode_config('fast')
    
    solver = ReverseLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=mode['L1'],
        L2=mode['L2'],
        verbose=False
    )
    
    target_arrival_time = 9 * 60 * 10
    alpha = 0.95
    
    print(f"  测试多个不同的OD对")
    print(f"  目标到达: {time_to_string(target_arrival_time)}, α={alpha}\n")
    
    # 测试5对不同的OD
    num_tests = 5
    success_count = 0
    results = []
    
    for i in range(num_tests):
        seed = 5000 + i
        origin, destination = select_od_pair(node_to_index)
        
        print(f"  测试 {i+1}/{num_tests} (seed={seed}): {origin}→{destination}...", 
              end='', flush=True)
        
        result = solver.solve(
            origin, destination, target_arrival_time, alpha,
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            success_count += 1
            # ✅ 修改：返回完整数据
            results.append({
                'od': (origin, destination),
                'origin': origin,  # ← 新增
                'destination': destination,  # ← 新增
                'latest_dep': result['latest_departure_time'],
                'expected_dep': result['expected_departure_time'],  # ← 新增
                'reserved': result['reserved_time'],
                'path': result['path'],  # ← 新增
                'path_length': len(result['path']),
                'target_arrival': target_arrival_time,  # ← 新增
                'alpha': alpha,  # ← 新增
                'distribution': result['distribution']  # ← 新增（用于可视化）
            })
            print(f" ✓ 预留={result['reserved_time']/10:.1f}分, 路径={len(result['path'])}节点")
        else:
            print(f" ✗ 失败")
    
    # 验证
    print(f"\n{'─'*70}")
    print(f"测试5验证")
    print(f"{'─'*70}")
    
    success_rate = success_count / num_tests * 100
    print(f"  成功率: {success_count}/{num_tests} ({success_rate:.1f}%)")
    
    assert success_count >= num_tests * 0.6, \
        f"❌ 成功率太低: {success_rate:.1f}%"
    print(f"  ✓ 成功率达标 (≥60%)")
    
    if results:
        print(f"\n  结果统计:")
        reserved_times = [r['reserved']/10 for r in results]
        path_lengths = [r['path_length'] for r in results]
        print(f"    预留时间: 均值={np.mean(reserved_times):.1f}分, "
              f"标准差={np.std(reserved_times):.1f}分")
        print(f"    路径长度: 均值={np.mean(path_lengths):.1f}, "
              f"范围=[{min(path_lengths)}, {max(path_lengths)}]")
    
    print(f"\n  🎉 测试5通过！")
    print(f"{'='*70}\n")
    
    return results  # ← 返回完整结果


# ═══════════════════════════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════════════════════════

def run_all_tests():
    """运行所有测试"""
    print(f"\n{'='*70}")
    print(f"反向求解器测试套件（优化版）")
    print(f"{'='*70}")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  配置: L1={config.REVERSE_L1}, L2={config.REVERSE_L2}")
    print(f"  优化: 全局加载数据 + 独立随机种子")
    print(f"{'='*70}\n")
    
    # 预先加载数据
    load_data_once()
    
    start_time = time_module.time()
    
    try:
        # 测试1: 基本求解
        test_1_basic_solve()
        
        # 测试2: α敏感性（完整版）
        test_2_alpha_sensitivity()
        
        # 测试3: 性能
        test_3_performance()
        
        # 测试4: 时间一致性
        test_4_time_consistency()
        
        # 测试5: 多OD对
        test_5_multiple_od_pairs()
        
        total_time = time_module.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"所有测试完成！✓")
        print(f"{'='*70}")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  状态: 全部通过 ✓")
        print(f"  数据加载: 仅一次（优化）")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"测试失败！✗")
        print(f"{'='*70}")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        return False

# 在文件末尾添加
from visualization_generator import generate_html_with_svg

def run_all_tests_with_visualization():
    """运行所有测试并生成可视化（修改版）"""
    print(f"\n{'='*70}")
    print(f"反向求解器测试套件（带可视化）")
    print(f"{'='*70}\n")
    
    # 预先加载数据
    load_data_once()
    G, _, _, _, _, _ = get_data()
    
    start_time = time_module.time()
    
    # 存储所有结果
    results_all = {}
    
    try:
        # 运行测试1
        print("运行测试1...")
        results_all['test1'] = test_1_basic_solve()
        
        # 运行测试2（增强版）
        print("运行测试2...")
        results_all['test2'] = test_2_alpha_sensitivity()
        
        # 运行测试3
        print("运行测试3...")
        results_all['test3'] = []  # 可选
        
        # 运行测试5（修改版）
        print("运行测试5...")
        results_all['test5'] = test_5_multiple_od_pairs()
        
        total_time = time_module.time() - start_time
        
        print(f"\n所有测试完成！总耗时: {total_time:.2f}秒")
        
        # ✅ 生成可视化
        print("\n生成HTML+SVG可视化...")
        from visualization_generator import generate_html_with_svg
        generate_html_with_svg(G, results_all, 'reverse_solver_visualization.html')
        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
# ═══════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # 验证配置
    print("验证配置...")
    config.validate_config()
    print()
    
    if len(sys.argv) > 1:
        # 预先加载数据
        load_data_once()
        
        # 运行指定测试
        test_name = sys.argv[1]
        if test_name == '1':
            test_1_basic_solve()
        elif test_name == '2':
            test_2_alpha_sensitivity()
        elif test_name == '3':
            test_3_performance()
        elif test_name == '4':
            test_4_time_consistency()
        elif test_name == '5':
            test_5_multiple_od_pairs()
        else:
            print(f"未知测试: {test_name}")
            print(f"可用测试: 1, 2, 3, 4, 5")
    else:
        # 运行所有测试并生成可视化
        success = run_all_tests_with_visualization()
        sys.exit(0 if success else 1)