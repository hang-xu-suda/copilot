"""
反向Label-Setting求解器 - 严格按照伪代码实现（完整版）
包含详细的概率计算说明
"""

import numpy as np
import heapq
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AlphaDiscreteDistribution:
    """α离散分布类"""
    values: List[float]
    L1: int
    
    def __init__(self, values: List[float], L1: int):
        if len(values) != L1:
            raise ValueError(f"期望{L1}个值，实际得到{len(values)}个")
        self.L1 = L1
        self.values = sorted(values)
    
    def get_quantile(self, alpha: float) -> float:
        """获取α分位数（线性插值）"""
        if alpha <= 0:
            return self.values[0]
        if alpha >= 1:
            return self.values[-1]
        
        index = alpha * (self.L1 - 1)
        lower_idx = int(np.floor(index))
        upper_idx = min(lower_idx + 1, self.L1 - 1)
        
        if lower_idx == upper_idx:
            return self.values[lower_idx]
        
        weight = index - lower_idx
        return self.values[lower_idx] * (1 - weight) + self.values[upper_idx] * weight
    
    def get_mean(self) -> float:
        return np.mean(self.values)
    
    def get_std(self) -> float:
        return np.std(self.values)
    
    def get_median(self) -> float:
        return np.median(self.values)
    
    def get_variance(self) -> float:
        """获取方差"""
        return np.var(self.values)
    
    def reverse_convolve(self,
                    get_link_dist_func,
                    predecessor: int,
                    current: int,
                    time_intervals_per_day: int,
                    L2: int) -> 'AlphaDiscreteDistribution':
        """
        反向卷积（基于精确概率计算，不采样）
        
        核心思想：
        1.遍历所有可能的出发时间 t_dep
        2.对每个 t_dep，计算其对应的时间片 slot_dep
        3.对每个到达时间 t_arr，计算所需旅行时间 k = t_arr - t_dep
        4.从 D_uv(slot_dep) 直接查询 P(k)，而非采样
        5.累加：P(t_dep) += P(t_arr) × P(k | slot_dep)
        
        Args:
            get_link_dist_func: 获取链路分布的函数
            predecessor: 前驱节点u
            current: 当前节点v
            time_intervals_per_day: 每天时间片数
            L2: 未使用（保持接口兼容）
            
        Returns:
            出发时间分布 A(u)
        """
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤1: 获取可用时间片和旅行时间范围
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        available_slots = self._get_available_slots(get_link_dist_func, predecessor, current)
        
        if not available_slots:
            raise ValueError(f"边({predecessor}, {current})没有链路分布数据")
        
        # 估算旅行时间范围（用于确定出发时间搜索范围）
        min_travel = float('inf')
        max_travel = 0
        
        for slot in available_slots:
            D_slot = get_link_dist_func(predecessor, current, slot)
            if D_slot and D_slot.times:
                min_travel = min(min_travel, min(D_slot.times))
                max_travel = max(max_travel, max(D_slot.times))
        
        if min_travel == float('inf'):
            raise ValueError(f"无法获取边({predecessor}, {current})的旅行时间范围")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2: 确定出发时间搜索范围
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 从所有到达时间推导出发时间的可能范围
        min_arrival = min(self.values)
        max_arrival = max(self.values)
        
        min_departure = min_arrival - max_travel
        max_departure = max_arrival - min_travel
        
        # 离散化出发时间：步长为10（1分钟）
        # 可以根据需要调整步长，越小越精确但计算量越大
        step = 1  # 0.1分钟单位 × 10 = 1分钟
        
        # 生成候选出发时间
        candidate_departures = np.arange(
            int(min_departure),
            int(max_departure) + step,
            step
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤3: 构建到达时间的概率分布
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 将 self.values (L1个样本) 转换为概率分布
        arrival_probs = {}
        for t_arr in self.values:
            if t_arr not in arrival_probs:
                arrival_probs[t_arr] = 0
            arrival_probs[t_arr] += 1.0 / self.L1
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤4: 对每个候选出发时间，计算其概率
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        departure_probs = {}
        
        for t_dep in candidate_departures:
            # 确定出发时间对应的时间片
            slot_dep = int(t_dep / 10) % time_intervals_per_day
            
            # 获取该时间片的路段分布
            D_slot = get_link_dist_func(predecessor, current, slot_dep)
            
            if D_slot is None:
                # 如果精确时间片没有分布，尝试最近的时间片
                nearest_slot = self._find_nearest_slot(
                    slot_dep, available_slots, time_intervals_per_day
                )
                D_slot = get_link_dist_func(predecessor, current, nearest_slot)
            
            if D_slot is None:
                continue
            
            # 计算 P(t_dep)
            prob_t_dep = 0.0
            
            for t_arr, prob_arr in arrival_probs.items():
                # 计算所需旅行时间
                required_travel_time = t_arr - t_dep
                
                # 查询该旅行时间的概率
                prob_travel = D_slot.get_probability(required_travel_time)
                
                if prob_travel > 0:
                    # P(t_dep) += P(t_arr) × P(travel_time = t_arr - t_dep | slot_dep)
                    prob_t_dep += prob_arr * prob_travel
            
            if prob_t_dep > 0:
                departure_probs[t_dep] = prob_t_dep
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤5: 归一化并构造新分布
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if not departure_probs:
            raise ValueError(f"反向卷积失败：无有效出发时间概率")
        
        # 归一化
        total_prob = sum(departure_probs.values())
        if total_prob <= 0:
            raise ValueError(f"反向卷积失败：总概率为0")
        
        for t in departure_probs:
            departure_probs[t] /= total_prob
        
        # 转换为数组
        times = np.array(sorted(departure_probs.keys()))
        probs = np.array([departure_probs[t] for t in times])
        
        # 重新归一化（数值稳定性）
        probs = probs / probs.sum()
        
        # 方法1: 按概率加权采样L1个值
        sampled_indices = np.random.choice(len(times), size=self.L1, replace=True, p=probs)
        sampled_times = times[sampled_indices]
        
        # 方法2（可选）: 按累积分位数选择L1个确定性代表值
        # cdf = np.cumsum(probs)
        # quantiles = np.linspace(1/(self.L1+1), self.L1/(self.L1+1), self.L1)
        # sampled_times = np.interp(quantiles, cdf, times)
        
        sampled_times.sort()
        
        return AlphaDiscreteDistribution(sampled_times.tolist(), self.L1)


    def _find_nearest_slot(self, target_slot: int, available_slots: List[int],
                        time_intervals_per_day: int) -> int:
        """找到最近的时间片"""
        min_dist = float('inf')
        best_slot = available_slots[0]
        
        for slot in available_slots:
            dist = abs(slot - target_slot)
            cyclic_dist = min(dist, time_intervals_per_day - dist)
            
            if cyclic_dist < min_dist:
                min_dist = cyclic_dist
                best_slot = slot
        
        return best_slot
    
    def _get_available_slots(self, get_link_dist_func, u: int, v: int) -> List[int]:
        if not hasattr(AlphaDiscreteDistribution, '_slot_cache'):
            AlphaDiscreteDistribution._slot_cache = {}
        cache_key = (u, v)
        if cache_key in AlphaDiscreteDistribution._slot_cache:
            return AlphaDiscreteDistribution._slot_cache[cache_key]
        available = []
        try:
            link_distributions = get_link_dist_func.__self__.link_distributions
            for (link_u, link_v, slot) in link_distributions.keys():
                if link_u == u and link_v == v:
                    available.append(slot)
        except AttributeError:
            raise ValueError("无法访问链路分布数据")
        result = sorted(set(available))
        AlphaDiscreteDistribution._slot_cache[cache_key] = result
        return result
    
    def _get_slots_in_range(self, slot_min: int, slot_max: int,
                           available_slots: List[int],
                           time_intervals_per_day: int) -> List[int]:
        """获取范围[slot_min, slot_max]内的候选时间片"""
        candidate_slots = []
        
        if slot_min <= slot_max:
            # 正常范围
            for slot in available_slots:
                if slot_min <= slot <= slot_max:
                    candidate_slots.append(slot)
        else:
            # 跨天范围
            for slot in available_slots:
                if slot >= slot_min or slot <= slot_max:
                    candidate_slots.append(slot)
        
        return sorted(candidate_slots)


@dataclass
class LinkTimeDistribution:
    """路段旅行时间分布"""
    time_prob: Dict[int, float]
    times: List[int]
    cdf: List[float]
    time_slot: int
    
    def __init__(self, time_prob_dict: Dict[int, float], time_slot: int = None):
        if not time_prob_dict:
            raise ValueError("链路分布不能为空")
        
        total_prob = sum(time_prob_dict.values())
        self.time_prob = {t: p/total_prob for t, p in time_prob_dict.items()}
        self.time_slot = time_slot
        
        sorted_times = sorted(self.time_prob.keys())
        self.times = sorted_times
        
        cumulative = 0.0
        self.cdf = []
        for t in sorted_times:
            cumulative += self.time_prob[t]
            self.cdf.append(cumulative)
    
    def sample_L2_times(self, reference_time: int, L2: int) -> List[int]:
        """采样L2个旅行时间（逆CDF方法）"""
        samples = []
        for i in range(1, L2 + 1):
            quantile = i / (L2 + 1)
            sample = self._inverse_cdf(quantile)
            samples.append(sample)
        return sorted(samples)
    
    def _inverse_cdf(self, quantile: float) -> int:
        """逆CDF（线性插值）"""
        if quantile <= 0:
            return self.times[0]
        if quantile >= 1:
            return self.times[-1]
        
        for i, cdf_val in enumerate(self.cdf):
            if cdf_val >= quantile:
                if i == 0:
                    return self.times[0]
                
                lower_cdf = self.cdf[i-1] if i > 0 else 0
                upper_cdf = cdf_val
                lower_time = self.times[i-1] if i > 0 else self.times[0]
                upper_time = self.times[i]
                
                if upper_cdf > lower_cdf:
                    weight = (quantile - lower_cdf) / (upper_cdf - lower_cdf)
                else:
                    weight = 0.5
                
                return int(round(lower_time + weight * (upper_time - lower_time)))
        
        return self.times[-1]
    
    def get_probability(self, travel_time: float) -> float:
        """
        获取指定旅行时间的概率
        
        Args:
            travel_time: 旅行时间（0.1分钟单位）
            
        Returns:
            该旅行时间的概率（如果不在支持集中，可选择插值或返回0）
        """
        # 转换为整数（与存储的键匹配）
        travel_time_int = int(round(travel_time))
        
        # 精确匹配
        if travel_time_int in self.time_prob:
            return self.time_prob[travel_time_int]
        
        # 可选：线性插值（如果需要更平滑的结果）
        if self.times:
            min_time = self.times[0]
            max_time = self.times[-1]
            
            if travel_time_int < min_time or travel_time_int > max_time:
                return 0.0
            
            # 找到相邻的两个点
            for i in range(len(self.times) - 1):
                if self.times[i] <= travel_time_int <= self.times[i+1]:
                    # 线性插值
                    t_lower = self.times[i]
                    t_upper = self.times[i+1]
                    p_lower = self.time_prob[t_lower]
                    p_upper = self.time_prob[t_upper]
                    
                    if t_upper == t_lower:
                        return p_lower
                    
                    weight = (travel_time_int - t_lower) / (t_upper - t_lower)
                    return p_lower * (1 - weight) + p_upper * weight
        
        return 0.0

    def get_mean(self) -> float:
        return sum(t * p for t, p in self.time_prob.items())
    
    def get_std(self) -> float:
        mean = self.get_mean()
        variance = sum(p * (t - mean)**2 for t, p in self.time_prob.items())
        return np.sqrt(variance)


@dataclass
class ReverseLabel:
    """反向搜索标签"""
    node_id: int
    distribution: AlphaDiscreteDistribution
    path: List[int]
    cost: float
    quantile_cache: Dict[float, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # 预计算常用分位数
        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            self.quantile_cache[q] = self.distribution.get_quantile(q)
        
        # 预计算均值和方差
        self.mean_cache = self.distribution.get_mean()
        self.variance_cache = self.distribution.get_variance()
    
    def __lt__(self, other):
        return self.cost > other.cost
    
    def get_cached_quantile(self, alpha: float) -> float:
        if alpha in self.quantile_cache:
            return self.quantile_cache[alpha]
        value = self.distribution.get_quantile(alpha)
        self.quantile_cache[alpha] = value
        return value
    
    def dominates_weak(self, other: 'ReverseLabel', alpha: float, epsilon: float = 1e-6) -> bool:
        """
        弱支配检查（保守剪枝）
        
        仅在以下情况支配：
        1.主目标显著更优 + 次要目标不更差
        2.主目标不更差 + 两个次要目标都显著更优
        """
        if self.node_id != other.node_id:
            return False
        
        self_q = self.distribution.get_quantile(1 - alpha)
        other_q = other.distribution.get_quantile(1 - alpha)
        
        # 策略A：主目标显著更优
        if self_q > other_q + epsilon:
            # 次要目标必须不更差
            if (self.mean_cache >= other.mean_cache - epsilon and
                self.variance_cache <= other.variance_cache + epsilon):
                return True
        
        # 策略B：主目标相近，但次要目标显著更优
        if abs(self_q - other_q) <= epsilon:
            # 两个次要目标都显著更优
            if (self.mean_cache > other.mean_cache + epsilon and
                self.variance_cache < other.variance_cache - epsilon):
                return True
            
            # 或者：均值显著更优 + 方差不更差
            if (self.mean_cache > other.mean_cache + epsilon and
                self.variance_cache <= other.variance_cache + epsilon):
                return True
            
            # 或者：方差显著更优 + 均值不更差  
            if (self.variance_cache < other.variance_cache - epsilon and
                self.mean_cache >= other.mean_cache - epsilon):
                return True
        
        return False
    
    def dominates(self, other: 'ReverseLabel', alpha: float, epsilon: float = 1e-6) -> bool:
        """统一接口"""
        return self.dominates_weak(other, alpha, epsilon)



# ═══════════════════════════════════════════════════════════════════
# 反向求解器主类
# ═══════════════════════════════════════════════════════════════════

class ReverseLabelSettingSolver:
    """反向Label-Setting求解器（完整版）"""
    
    def __init__(self, G, sparse_data, node_to_index, scenario_dates,
                 scenario_probs, time_intervals_per_day,
                 L1: int = 50, L2: int = 10,
                 verbose: bool = False,
                 max_labels_per_node: int = 20):
        """初始化"""
        self.G = G
        self.sparse_data = sparse_data
        self.node_to_index = node_to_index
        self.index_to_node = {v: k for k, v in node_to_index.items()}
        self.scenario_dates = scenario_dates
        self.scenario_probs = scenario_probs
        self.time_intervals_per_day = time_intervals_per_day
        self.n_scenarios = len(scenario_dates)
        
        self.L1 = L1
        self.L2 = L2
        self.verbose = verbose

        self.max_labels_per_node = max_labels_per_node
        
        print(f"\n{'='*70}")
        print(f"初始化反向Label-Setting求解器（完整版）")
        print(f"{'='*70}")
        print(f"  算法: 反向Label-Setting with 概率权重")
        print(f"  问题: 预留时间预算")
        print(f"  参数: L1={L1}, L2={L2}")
        print(f"  详细输出: {'开启' if verbose else '关闭'}")
        
        # 构建邻接表
        self.adj_list = defaultdict(list)
        self.reverse_adj_list = defaultdict(list)
        self._build_adjacency_lists()
        
        # 预计算链路分布
        self.link_distributions = {}
        self._precompute_link_distributions()
        
        # 统计信息
        self.stats = defaultdict(int)
        self.origin_labels_history = []
        
        print(f"\n✓ 初始化完成")
        print(f"{'='*70}\n")
    
    def _build_adjacency_lists(self):
        """构建邻接表"""
        print(f"  [1/2] 构建邻接表...")
        start_time = time.time()
        
        edges_set = set()
        for (scenario_idx, time_idx, from_idx, to_idx) in self.sparse_data.keys():
            if scenario_idx < self.n_scenarios:
                from_node = self.index_to_node[from_idx]
                to_node = self.index_to_node[to_idx]
                edges_set.add((from_node, to_node))
        
        for from_node, to_node in edges_set:
            self.adj_list[from_node].append(to_node)
            self.reverse_adj_list[to_node].append(from_node)
        
        elapsed = time.time() - start_time
        print(f"      ✓ 完成 (用时 {elapsed:.2f}s) - {len(edges_set):,} 条边")
    
    def _precompute_link_distributions(self):
        """预计算链路分布"""
        print(f"  [2/2] 预计算链路分布...")
        start_time = time.time()
        
        link_time_data = defaultdict(list)
        
        for (scenario_idx, time_idx, from_idx, to_idx), travel_time_minutes in self.sparse_data.items():
            if scenario_idx >= self.n_scenarios:
                continue
            
            from_node = self.index_to_node[from_idx]
            to_node = self.index_to_node[to_idx]
            travel_time_01min = int(travel_time_minutes * 10)
            
            link_time_data[(from_node, to_node, time_idx)].append(travel_time_01min)
        
        distribution_count = 0
        for (u, v, t), times in link_time_data.items():
            time_counts = defaultdict(int)
            for time_val in times:
                time_counts[time_val] += 1
            
            total = len(times)
            time_prob = {time_val: count/total for time_val, count in time_counts.items()}
            
            try:
                self.link_distributions[(u, v, t)] = LinkTimeDistribution(time_prob, time_slot=t)
                distribution_count += 1
            except ValueError:
                continue
        
        elapsed = time.time() - start_time
        print(f"      ✓ 完成 (用时 {elapsed:.2f}s) - {distribution_count:,} 个分布")
    
    def _get_link_distribution_at_slot(self, u: int, v: int, slot: int) -> Optional[LinkTimeDistribution]:
        """获取指定出发时间片的链路分布"""
        if (u, v, slot) in self.link_distributions:
            return self.link_distributions[(u, v, slot)]
        
        # 容差匹配
        tolerance = 5
        candidates = []
        
        for (link_u, link_v, link_t) in self.link_distributions.keys():
            if link_u == u and link_v == v:
                diff = abs(link_t - slot)
                cyclic_diff = min(diff, self.time_intervals_per_day - diff)
                
                if cyclic_diff <= tolerance:
                    candidates.append((link_t, cyclic_diff))
        
        if candidates:
            best_slot = min(candidates, key=lambda x: x[1])[0]
            return self.link_distributions[(u, v, best_slot)]
        
        return None
    
    def solve_k_paths(self, origin: int, destination: int, target_arrival_time: int,
                     alpha: float, K: int = 10, max_labels:  int = 100000,
                     print_interval: int = 100) -> Dict: 
        """
        K-Paths 反向求解
        
        Args: 
            origin: 起点
            destination: 终点
            target_arrival_time: 目标到达时间
            alpha: 可靠性参数
            K: 候选路径数量
            max_labels: 最大标签数
            print_interval: 打印间隔
        
        Returns:
            包含K条候选路径的结果字典
        """
        
        print(f"\n{'='*70}")
        print(f"反向Label-Setting求解（K-Paths版本）")
        print(f"{'='*70}")
        print(f"  起点: {origin}")
        print(f"  终点: {destination}")
        print(f"  目标到达:  {target_arrival_time/10:.1f}分")
        print(f"  可靠性: α={alpha*100:.1f}%")
        print(f"  候选路径数:  K={K}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # 初始化
        open_labels = []
        node_labels = defaultdict(list)
        origin_candidates = []  # ✅ 存储所有到达起点的候选标签
        
        # 初始标签
        init_dist = AlphaDiscreteDistribution([target_arrival_time] * self.L1, self.L1)
        init_label = ReverseLabel(destination, init_dist, [destination], target_arrival_time)
        
        heapq.heappush(open_labels, init_label)
        node_labels[destination].append(init_label)
        self.stats = defaultdict(int)
        self.stats['labels_generated'] = 1
        
        print(f"开始搜索 K={K} 条候选路径...\n")
        
        iteration = 0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 主循环：找到K条到达起点的路径
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        while open_labels and self.stats['labels_generated'] < max_labels:
            iteration += 1
            current_label = heapq.heappop(open_labels)
            
            if self.verbose and (iteration % print_interval == 0 or iteration <= 5):
                print(f"  迭代#{iteration}: 节点{current_label.node_id}, "
                      f"cost={current_label.cost/10:.1f}分, "
                      f"候选数={len(origin_candidates)}")
            
            # ✅ 到达起点：保存为候选路径
            if current_label.node_id == origin:
                latest_departure = current_label.distribution.get_quantile(1 - alpha)
                expected_departure = current_label.mean_cache
                
                # 保存候选路径信息
                candidate_info = {
                    'iteration': iteration,
                    'path': list(reversed(current_label. path)),
                    'distribution': current_label.distribution,
                    'latest_departure': latest_departure,
                    'expected_departure': expected_departure,
                    'median_departure': current_label.distribution.get_median(),
                    'std_departure': np.sqrt(current_label.variance_cache),
                    'variance': current_label.variance_cache,
                    'label': current_label,
                    'alpha': alpha,
                    'rank': None,
                    'is_best':  False  # ← 添加这个字段
                }
                
                origin_candidates.append(candidate_info)
                
                print(f"  🎯 找到候选路径#{len(origin_candidates)}  迭代#{iteration}, "
                      f"Q_{{1-α}}={latest_departure/10:.1f}分, "
                      f"Mean={expected_departure/10:.1f}分, "
                      f"路径长度={len(current_label.path)}")
                
                # ✅ 找到K条路径后继续搜索（确保探索充分）
                if len(origin_candidates) >= K:
                    # 可以选择：
                    # 选项A：立即停止（快速）
                    # 选项B：继续搜索一段时间（更全面）
                    
                    # 这里使用选项B：继续搜索，但有上限
                    if len(origin_candidates) >= K * 2:  # 找到2K条后停止
                        print(f"\n  ✓ 已找到 {len(origin_candidates)} 条候选路径，停止搜索\n")
                        break
                
                # 继续搜索其他路径
                continue
            
            # 支配性检查（较宽松，保留多样性）
            if self._is_dominated(current_label, node_labels[current_label.node_id], alpha):
                self.stats['labels_dominated'] += 1
                continue
            
            self.stats['labels_extended'] += 1
            
            # 反向扩展
            if current_label.node_id not in self.reverse_adj_list:
                continue
            
            for predecessor in self.reverse_adj_list[current_label.node_id]:
                if predecessor in current_label.path:
                    continue
                
                # 反向卷积
                try:
                    def get_link_dist(u, v, slot):
                        return self._get_link_distribution_at_slot(u, v, slot)
                    
                    get_link_dist.__self__ = self
                    
                    new_dist = current_label.distribution.reverse_convolve(
                        get_link_dist_func=get_link_dist,
                        predecessor=predecessor,
                        current=current_label.node_id,
                        time_intervals_per_day=self.time_intervals_per_day,
                        L2=self.L2
                    )
                    
                    self.stats['convolutions'] += 1
                    
                except Exception as e:
                    if self.verbose and iteration <= 10:
                        print(f"      ⚠ 卷积失败: {e}")
                    continue
                
                new_cost = new_dist.get_quantile(1 - alpha)
                new_label = ReverseLabel(predecessor, new_dist, 
                                        current_label.path + [predecessor], new_cost)
                
                self.stats['labels_generated'] += 1
                
                # 支配性剪枝
                if self._is_dominated(new_label, node_labels[predecessor], alpha):
                    self.stats['labels_dominated'] += 1
                    continue
                
                # 反向剪枝
                original_count = len(node_labels[predecessor])
                node_labels[predecessor] = [
                    old for old in node_labels[predecessor]
                    if not new_label.dominates_weak(old, alpha)
                ]
                self.stats['labels_dominated'] += (original_count - len(node_labels[predecessor]))
                
                node_labels[predecessor].append(new_label)
                node_labels[predecessor] = self._prune_labels(node_labels[predecessor], alpha)
                heapq.heappush(open_labels, new_label)
            
            # 进度显示
            if not self.verbose and iteration % 100 == 0:
                print(f"  进度: 迭代#{iteration}, 生成{self.stats['labels_generated']: ,}, "
                      f"候选{len(origin_candidates)}, "
                      f"剪枝{self.stats['labels_dominated']:,}", end='\r')
        
        total_time = time.time() - start_time
        
        print(f"\n\n{'='*70}")
        print(f"搜索完成")
        print(f"{'='*70}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2：对K条候选路径排序
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if not origin_candidates:
            print(f"✗ 未找到到达起点的路径")
            return {
                'success': False,
                'total_time': total_time,
                'iterations': iteration,
                'stats':  dict(self.stats),
                'num_candidates': 0
            }
        
        print(f"\n找到 {len(origin_candidates)} 条候选路径")
        print(f"开始排序和比较...\n")
        
        # ✅ 多目标排序：主要Q_{1-α}，次要Mean，再次要-Var
        def rank_score(candidate):
            return (
                candidate['latest_departure'],      # 主目标：Q_{1-α}（越大越好）
                candidate['expected_departure'],    # 次要：均值（越大越好）
                -candidate['variance']              # 再次要：方差（越小越好）
            )
        
        # 排序：从最优到最差
        sorted_candidates = sorted(origin_candidates, key=rank_score, reverse=True)
        
        # 设置排名
        for rank, candidate in enumerate(sorted_candidates, 1):
            candidate['rank'] = rank
            candidate['is_best'] = (rank == 1)  # ← 排名第1的标记为最优
        
        # 取前K条
        top_k_candidates = sorted_candidates[:K]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤3：输出结果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        best_candidate = top_k_candidates[0]
        
        print(f"{'='*70}")
        print(f"Top-{len(top_k_candidates)} 候选路径对比")
        print(f"{'='*70}\n")
        
        print(f"{'排名':<6} {'Q_{{1-α}}(分)':<15} {'Mean(分)':<15} {'Std(分)':<12} {'路径长度':<10}")
        print(f"{'-'*70}")
        
        for candidate in top_k_candidates: 
            print(f"{candidate['rank']:<6} "
                  f"{candidate['latest_departure']/10:<15.1f} "
                  f"{candidate['expected_departure']/10:<15.1f} "
                  f"{candidate['std_departure']/10:<12.2f} "
                  f"{len(candidate['path']):<10}")
        
        print(f"\n{'='*70}")
        print(f"✓ 最优路径（排名#1）")
        print(f"{'='*70}")
        print(f"\n  路径:  {self._format_path(best_candidate['path'])}")
        print(f"  长度: {len(best_candidate['path'])} 个节点")
        print(f"\n  时间:")
        print(f"    目标到达: {target_arrival_time/10:.1f}分")
        print(f"    最晚出发 (α={alpha}): {best_candidate['latest_departure']/10:.1f}分")
        print(f"    期望出发: {best_candidate['expected_departure']/10:.1f}分")
        print(f"    预留时间: {(target_arrival_time - best_candidate['latest_departure'])/10:.1f}分")
        print(f"    标准差:  {best_candidate['std_departure']/10:.2f}分")
        print(f"\n  性能:")
        print(f"    总耗时: {total_time:.2f}秒")
        print(f"    迭代次数: {iteration}")
        print(f"    候选路径数: {len(origin_candidates)}")
        print(f"    生成标签:  {self.stats['labels_generated']: ,}")
        print(f"    剪枝率: {self.stats['labels_dominated']/self.stats['labels_generated']*100:.1f}%")
        print(f"{'='*70}\n")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 构建返回结果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        result = {
            'success': True,
            # 最优路径信息
            'path': best_candidate['path'],
            'latest_departure_time': best_candidate['latest_departure'],
            'expected_departure_time': best_candidate['expected_departure'],
            'median_departure_time': best_candidate['median_departure'],
            'std_departure_time': best_candidate['std_departure'],
            'reserved_time': target_arrival_time - best_candidate['latest_departure'],
            'distribution': best_candidate['distribution'],
            
            # Top-K候选路径
            'top_k_candidates': top_k_candidates,
            'num_candidates': len(origin_candidates),
            'all_candidates': sorted_candidates,
            
            # 元信息
            'total_time':  total_time,
            'iterations': iteration,
            'alpha': alpha,
            'K':  K,
            'origin':  origin,
            'destination': destination,
            'target_arrival_time': target_arrival_time,
            'stats': dict(self.stats)
        }
        
        return result

    def solve(self, origin: int, destination: int, target_arrival_time: int,
            alpha: float, max_labels: int = 100000,
            print_interval: int = 100,
            save_all_paths: bool = True) -> Dict: 
        """
        标准solve接口（兼容原代码）
        
        内部调用solve_k_paths，K=5（返回5条候选路径）
        """
        result = self.solve_k_paths(
            origin, destination, target_arrival_time, alpha,
            K=5, max_labels=max_labels, print_interval=print_interval
        )
        
        # 添加all_paths用于兼容
        if save_all_paths and 'all_candidates' in result:
            result['all_paths'] = result['all_candidates']
            result['num_candidate_paths'] = result['num_candidates']
        
        return result
    
    def _is_dominated(self, label: ReverseLabel, existing_labels: List[ReverseLabel], 
                     alpha: float) -> bool:
        """
        支配性检查（保守版本）
        
        策略：
        1.如果节点标签数 < max_labels_per_node：不剪枝
        2.如果已满：检查是否被弱支配
        """
        # 策略1：保留多样性
        if len(existing_labels) < self.max_labels_per_node:
            # 只有被多个标签明确支配时才剪枝
            domination_count = 0
            for existing in existing_labels:
                if existing.dominates_weak(label, alpha):
                    domination_count += 1
            
            # 被2个以上标签支配才剪枝
            return domination_count >= 2
        
        # 策略2：标签数已满，使用弱支配
        for existing in existing_labels:
            if existing.dominates_weak(label, alpha):
                return True
        
        return False
    
    def _prune_labels(self, labels: List[ReverseLabel], alpha: float) -> List[ReverseLabel]:
        """
        当标签数超限时，移除最差的标签
        
        排序标准：
        1.Q_{1-α}（主要）
        2.Mean（次要）
        3.-Var（再次要）
        """
        if len(labels) <= self.max_labels_per_node:
            return labels
        
        # 多目标排序
        def label_score(label):
            q = label.distribution.get_quantile(1 - alpha)
            return (q, label.mean_cache, -label.variance_cache)
        
        # 保留最优的max_labels_per_node个
        sorted_labels = sorted(labels, key=label_score, reverse=True)
        return sorted_labels[:self.max_labels_per_node]

    def _format_path(self, path: List[int]) -> str:
        if len(path) <= 10:
            return ' → '.join(map(str, path))
        return f"{' → '.join(map(str, path[:5]))} → ...→ {' → '.join(map(str, path[-3:]))}"