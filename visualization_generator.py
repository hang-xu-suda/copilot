"""
反向求解器可视化生成器（完全重写版 - 增强测试5）
功能：
1.生成HTML+内嵌SVG可视化
2.支持SVG导出
3.测试2展示所有候选路径的概率分布对比
4.测试5完整展示多OD对结果（包含α、期望到达时间、完整路径节点）
"""

import json
import numpy as np
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def time_to_string(time_01min):
    """时间格式转换"""
    if time_01min is None:
        return "N/A"
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def get_path_coords(G, path):
    """获取路径坐标"""
    if not path:
        return []
    coords = []
    for node in path:
        if node in G.nodes:
            node_data = G.nodes[node]
            if 'y' in node_data and 'x' in node_data:
                coords.append([node_data['y'], node_data['x']])
    return coords


def format_path_nodes(path, max_display=20):
    """
    格式化路径节点显示
    
    Args:
        path: 节点列表
        max_display: 最多显示的节点数（超过则折叠）
    
    Returns:
        HTML格式的路径节点字符串
    """
    if not path:
        return "无路径"
    
    if len(path) <= max_display:
        # 全部显示
        nodes_html = ' → '.join([f'<span class="path-node">{node}</span>' for node in path])
        return f'<div class="path-full">{nodes_html}</div>'
    else:
        # 折叠显示
        visible_nodes = path[:10]
        hidden_nodes = path[10:-5]
        end_nodes = path[-5:]
        
        visible_html = ' → '.join([f'<span class="path-node">{node}</span>' for node in visible_nodes])
        hidden_html = ' → '.join([str(node) for node in hidden_nodes])
        end_html = ' → '.join([f'<span class="path-node">{node}</span>' for node in end_nodes])
        
        collapse_id = f"path_{hash(str(path))}"
        
        html = f'''
        <div class="path-container">
            <div class="path-visible">
                {visible_html}
                <button class="path-expand-btn" onclick="togglePath('{collapse_id}')">
                    ...(还有{len(hidden_nodes)}个节点) ...
                </button>
                {end_html}
            </div>
            <div id="{collapse_id}" class="path-hidden" style="display:none;">
                <div class="path-full-details">
                    <strong>完整路径 ({len(path)} 个节点):</strong><br>
                    {' → '.join([str(node) for node in path])}
                </div>
            </div>
        </div>
        '''
        return html


# ═══════════════════════════════════════════════════════════════════
# HTML生成辅助函数
# ═══════════════════════════════════════════════════════════════════

def generate_alpha_detail_options(test2_detailed):
    """生成α详细分析选项"""
    if not test2_detailed:
        return '<option value="">无详细数据</option>'
    
    options = ['<option value="">-- 选择α值 --</option>']
    for alpha in sorted(test2_detailed.keys()):
        alpha_val = float(alpha)
        options.append(f'<option value="{alpha_val}">{alpha_val:.2f}</option>')
    return '\n'.join(options)


def generate_alpha_summary_table(test2_results):
    """生成α敏感性汇总表"""
    if not test2_results:
        return '<tr><td colspan="5">无数据</td></tr>'
    
    rows = []
    for r in test2_results:
        rows.append(f'''<tr>
            <td>{r['alpha']:.2f}</td>
            <td>{time_to_string(r['latest_departure'])}</td>
            <td>{time_to_string(r['expected_departure'])}</td>
            <td>{r['reserved_time']/10:.1f}</td>
            <td>{r['path_length']}</td>
        </tr>''')
    return '\n'.join(rows)


def generate_od_options(test5_results):
    """生成OD选项"""
    if not test5_results:
        return '<option value="">无数据</option>'
    
    options = []
    for i, r in enumerate(test5_results):
        options.append(f'<option value="{i}">OD{i+1}: {r["origin"]} → {r["destination"]}</option>')
    return '\n'.join(options)


def generate_test5_table(test5_results):
    """生成测试5汇总表（增强版）"""
    if not test5_results:
        return '<tr><td colspan="9">无数据</td></tr>'
    
    rows = []
    for i, r in enumerate(test5_results, 1):
        # 构建行（避免f-string中的嵌套格式化）
        row = '<tr>'
        row += f'<td>{i}</td>'
        row += f'<td>{r["origin"]}</td>'
        row += f'<td>{r["destination"]}</td>'
        row += f'<td>{r["alpha"]}</td>'
        row += f'<td>{time_to_string(r.get("target_arrival"))}</td>'
        row += f'<td>{time_to_string(r["latest_dep"])}</td>'
        row += f'<td>{time_to_string(r.get("expected_dep"))}</td>'
        row += f'<td>{r["reserved"]:.1f}</td>'
        row += f'<td>{r["path_length"]}</td>'
        row += '</tr>'
        
        rows.append(row)
    
    return '\n'.join(rows)


# ═══════════════════════════════════════════════════════════════════
# 主生成函数
# ═══════════════════════════════════════════════════════════════════

def generate_html_with_svg(G, results_all_tests, output_file='reverse_solver_visualization.html'):
    """
    生成HTML+SVG可视化（完全重写版 - 增强测试5）
    
    Args:
        G: 路网图
        results_all_tests: 所有测试结果
        output_file: 输出文件路径
    """
    
    print(f"\n{'='*70}")
    print(f"生成HTML+SVG可视化")
    print(f"{'='*70}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 数据准备
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    test1_result = results_all_tests.get('test1', {})
    test2_data = results_all_tests.get('test2', {})
    test3_results = results_all_tests.get('test3', [])
    test5_results = results_all_tests.get('test5', [])
    
    # 处理测试2数据
    test2_results = []
    test2_detailed = {}
    
    if isinstance(test2_data, dict):
        test2_results = test2_data.get('all_results', [])
        test2_detailed = test2_data.get('detailed_results', {})
    elif isinstance(test2_data, list):
        test2_results = test2_data
    
    print(f"  测试1: {'成功' if test1_result.get('success') else '失败'}")
    print(f"  测试2: {len(test2_results)} 个α值, {len(test2_detailed)} 个详细分析")
    print(f"  测试5: {len(test5_results)} 个OD对")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 构建数据JSON
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    data_json = {
        'test1': {},
        'test2': {
            'summary': [],
            'detailed': {}
        },
        'test3': test3_results,
        'test5': []
    }
    
    # 处理测试1
    if test1_result.get('success'):
        path_coords = get_path_coords(G, test1_result['path'])
        if path_coords:
            center_lat = sum(c[0] for c in path_coords) / len(path_coords)
            center_lon = sum(c[1] for c in path_coords) / len(path_coords)
            data_json['test1'] = {
                'success': True,
                'origin': test1_result['path'][0],
                'destination': test1_result['path'][-1],
                'arrival_time': time_to_string(test1_result.get('target_arrival_time')),
                'departure_time': time_to_string(test1_result['latest_departure_time']),
                'expected_departure_time': time_to_string(test1_result.get('expected_departure_time')),
                'reserved_time': test1_result['reserved_time'] / 10,
                'path_length': len(test1_result['path']),
                'path': test1_result['path'],
                'path_coords': path_coords,
                'center_lat': center_lat,
                'center_lon': center_lon
            }
    
    # 处理测试2汇总
    for r in test2_results:
        path_coords = get_path_coords(G, r['path'])
        data_json['test2']['summary'].append({
            'alpha': float(r['alpha']),
            'latest_departure': float(r['latest_departure']),
            'latest_departure_str': time_to_string(r['latest_departure']),
            'expected_departure': float(r['expected_departure']),
            'expected_departure_str': time_to_string(r['expected_departure']),
            'reserved_time': float(r['reserved_time']) / 10,
            'path_length': int(r['path_length']),
            'path': r['path'],
            'path_coords': path_coords
        })
    

    # 处理测试2详细数据
    for alpha_key, detailed_result in test2_detailed.items():
        alpha = float(alpha_key)
        if 'all_paths' in detailed_result: 
            # 构建所有候选路径的分布数据
            all_paths_data = []
            for path_info in detailed_result['all_paths']:
                all_paths_data.append({
                    'values': path_info['distribution']. values,
                    'is_best': path_info. get('is_best', False),  # ← 修复：使用get，默认False
                    'path_length': len(path_info['path']),
                    'latest_departure': float(path_info['latest_departure']),
                    'expected_departure': float(path_info['expected_departure']),
                    'std_departure': float(path_info['std_departure'])
                })
            
            data_json['test2']['detailed'][str(alpha)] = {
                'alpha': alpha,
                'num_candidates': int(detailed_result['num_candidates']),
                'all_paths': all_paths_data,
                'best_path_coords': get_path_coords(G, detailed_result['path'])
            }
        
        # ✅ 新增：处理K-Paths版本的数据结构
        elif 'top_k_candidates' in detailed_result:
            # K-Paths版本返回的数据
            all_paths_data = []
            for candidate in detailed_result['all_candidates']:
                all_paths_data.append({
                    'values': candidate['distribution'].values,
                    'is_best': candidate.get('rank', 999) == 1,  # ← 排名第1的是最优
                    'path_length': len(candidate['path']),
                    'latest_departure': float(candidate['latest_departure']),
                    'expected_departure': float(candidate['expected_departure']),
                    'std_departure': float(candidate['std_departure'])
                })
            
            data_json['test2']['detailed'][str(alpha)] = {
                'alpha': alpha,
                'num_candidates':  int(detailed_result['num_candidates']),
                'all_paths': all_paths_data,
                'best_path_coords': get_path_coords(G, detailed_result['path'])
            }
    
    # 处理测试5（增强版 - 包含完整信息）
    for r in test5_results:
        path_coords = get_path_coords(G, r.get('path', []))
        data_json['test5'].append({
            'origin': r['origin'],
            'destination': r['destination'],
            'alpha': float(r.get('alpha', 0)),
            'target_arrival': float(r.get('target_arrival', 0)),
            'target_arrival_str': time_to_string(r.get('target_arrival')),
            'latest_departure': float(r['latest_dep']),
            'latest_departure_str': time_to_string(r['latest_dep']),
            'expected_departure': float(r.get('expected_dep', 0)),
            'expected_departure_str': time_to_string(r.get('expected_dep')),
            'reserved_time': float(r['reserved']) / 10,
            'path_length': int(r['path_length']),
            'path': r.get('path', []),
            'path_coords': path_coords
        })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 生成HTML内容
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>反向求解器测试结果 - 交互式可视化</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ color: #666; font-size: 1.1em; }}
        .nav-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-button {{
            padding: 15px 30px;
            border: none;
            background: white;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .tab-button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.15); }}
        .tab-button.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .card-title {{
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .info-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .info-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .info-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .selector-group {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        .selector-group label {{ font-weight: 600; margin-right: 10px; }}
        .selector-group select {{
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #667eea;
            font-size: 1em;
            min-width: 200px;
        }}
        .svg-container {{
            width: 100%;
            overflow-x: auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .svg-container svg {{ max-width: 100%; height: auto; }}
        .export-button {{
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin: 10px 5px;
        }}
        .export-button:hover {{ background: #5568d3; }}
        .map-container {{
            height: 500px;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .explanation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .explanation strong {{ color: #856404; }}
        
        /* 路径节点样式 */
        .path-container {{
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .path-node {{
            display: inline-block;
            padding: 3px 8px;
            background: #667eea;
            color: white;
            border-radius: 4px;
            margin: 2px;
            font-size: 0.9em;
            font-weight: 500;
        }}
        .path-expand-btn {{
            display: inline-block;
            padding: 3px 12px;
            background: #ffc107;
            color: #333;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            margin: 2px 5px;
            transition: background 0.3s;
        }}
        .path-expand-btn:hover {{
            background: #ffb300;
        }}
        .path-hidden {{
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        .path-full-details {{
            font-size: 0.9em;
            line-height: 1.8;
            color: #555;
        }}
        
        /* 详细信息卡片 */
        .detail-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid #667eea;
        }}
        .detail-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .detail-row:last-child {{
            border-bottom: none;
        }}
        .detail-label {{
            font-weight: 600;
            color: #555;
        }}
        .detail-value {{
            color: #667eea;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚗 反向求解器测试结果</h1>
            <p class="subtitle">预留时间预算问题 - Reverse Label-Setting Algorithm</p>
            <p class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="nav-tabs">
            <button class="tab-button active" onclick="showTab('overview')">📊 总览</button>
            <button class="tab-button" onclick="showTab('test1')">🎯 测试1</button>
            <button class="tab-button" onclick="showTab('test2')">📈 测试2</button>
            <button class="tab-button" onclick="showTab('test5')">🔄 测试5</button>
        </div>
        
        <!-- 总览 -->
        <div id="overview" class="tab-content active">
            <div class="card">
                <h2 class="card-title">测试总览</h2>
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">测试1: 基本求解</div>
                        <div class="info-value">{('✓' if test1_result.get('success') else '✗')}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试2: α点数</div>
                        <div class="info-value">{len(test2_results)}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试2: 详细分析</div>
                        <div class="info-value">{len(test2_detailed)}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试5: OD对数</div>
                        <div class="info-value">{len(test5_results)}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 测试1 -->
        <div id="test1" class="tab-content">
            <div class="card">
                <h2 class="card-title">测试1: 基本求解</h2>
                <div id="test1Info"></div>
                <div class="map-container" id="test1Map"></div>
            </div>
        </div>
        
        <!-- 测试2 -->
        <div id="test2" class="tab-content">
            <div class="card">
                <h2 class="card-title">测试2: α敏感性分析 - 路径分布对比</h2>
                
                <div class="explanation">
                    <strong>💡 说明：</strong>
                    下拉菜单中的α值是进行了详细分析的（保存了所有候选路径）。
                    选择后将展示该α值下所有候选路径的出发时间CDF分布对比图。
                    红色粗线是算法选择的最优路径，其他颜色是搜索过程中找到的候选路径。
                    通过对比可以理解为什么某条路径在该α下是最优的。
                </div>
                
                <div class="selector-group">
                    <label for="alphaDetailSelect">选择α值（查看详细路径分布）:</label>
                    <select id="alphaDetailSelect" onchange="updateAlphaDetailView()">
                        {generate_alpha_detail_options(test2_detailed)}
                    </select>
                </div>
                
                <div id="alphaDetailInfo"></div>
                <div class="svg-container" id="alphaDistributionChart"></div>
                <button class="export-button" onclick="exportSVG('alphaDistributionChart', 'alpha_distribution_comparison')">💾 导出分布对比图 (SVG)</button>
                
                <div class="map-container" id="test2Map"></div>
            </div>
            
            <div class="card">
                <h2 class="card-title">α敏感性汇总表</h2>
                <table>
                    <thead>
                        <tr>
                            <th>α值</th>
                            <th>最晚出发</th>
                            <th>期望出发</th>
                            <th>预留时间(分)</th>
                            <th>路径长度</th>
                        </tr>
                    </thead>
                    <tbody>
                        {generate_alpha_summary_table(test2_results)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- 测试5 -->
        <div id="test5" class="tab-content">
            <div class="card">
                <h2 class="card-title">测试5: 多OD对稳定性测试</h2>
                
                <div class="selector-group">
                    <label for="odSelect">选择OD对:</label>
                    <select id="odSelect" onchange="updateODView()">
                        {generate_od_options(test5_results)}
                    </select>
                </div>
                
                <div id="test5Info"></div>
                <div id="test5PathDetails"></div>
                <div class="map-container" id="test5Map"></div>
            </div>
            
            <div class="card">
                <h2 class="card-title">多OD对汇总表</h2>
                <table>
                    <thead>
                        <tr>
                            <th>编号</th>
                            <th>起点</th>
                            <th>终点</th>
                            <th>α值</th>
                            <th>目标到达</th>
                            <th>最晚出发</th>
                            <th>期望出发</th>
                            <th>预留(分)</th>
                            <th>路径长度</th>
                        </tr>
                    </thead>
                    <tbody>
                        {generate_test5_table(test5_results)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        // 数据
        const data = {json.dumps(data_json, ensure_ascii=False, cls=NumpyEncoder)};
        
        // 地图对象
        let maps = {{}};
        
        // ═══════════════════════════════════════════════════════════════════
        // 标签页切换
        // ═══════════════════════════════════════════════════════════════════
        
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'test1') initTest1();
            else if (tabName === 'test2') initTest2();
            else if (tabName === 'test5') initTest5();
        }}
        
        // ═══════════════════════════════════════════════════════════════════
        // 测试1初始化
        // ═══════════════════════════════════════════════════════════════════
        
        function initTest1() {{
            if (!data.test1.success) {{
                document.getElementById('test1Info').innerHTML = '<p>测试1未成功</p>';
                return;
            }}
            
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">起点</div>
                        <div class="info-value">${{data.test1.origin}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">终点</div>
                        <div class="info-value">${{data.test1.destination}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">目标到达</div>
                        <div class="info-value">${{data.test1.arrival_time}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">最晚出发</div>
                        <div class="info-value">${{data.test1.departure_time}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">期望出发</div>
                        <div class="info-value">${{data.test1.expected_departure_time}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">预留时间</div>
                        <div class="info-value">${{data.test1.reserved_time.toFixed(1)}}分</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">路径长度</div>
                        <div class="info-value">${{data.test1.path_length}}节点</div>
                    </div>
                </div>
            `;
            document.getElementById('test1Info').innerHTML = info;
            
            // 初始化地图
            if (!maps.test1 && data.test1.path_coords && data.test1.path_coords.length > 0) {{
                maps.test1 = L.map('test1Map').setView([data.test1.center_lat, data.test1.center_lon], 13);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(maps.test1);
                
                const polyline = L.polyline(data.test1.path_coords, {{color: '#667eea', weight: 5}}).addTo(maps.test1);
                maps.test1.fitBounds(polyline.getBounds());
                
                L.circleMarker(data.test1.path_coords[0], {{
                    radius: 10, fillColor: '#00ff00', color: '#006600',
                    weight: 2, fillOpacity: 0.8
                }}).addTo(maps.test1).bindPopup('起点');
                
                L.circleMarker(data.test1.path_coords[data.test1.path_coords.length - 1], {{
                    radius: 10, fillColor: '#ff0000', color: '#660000',
                    weight: 2, fillOpacity: 0.8
                }}).addTo(maps.test1).bindPopup('终点');
            }}
        }}
        
        // ═══════════════════════════════════════════════════════════════════
        // 测试2初始化
        // ═══════════════════════════════════════════════════════════════════
        
        function initTest2() {{
            // 等待用户选择α值
        }}
        
        function updateAlphaDetailView() {{
            const alphaSelect = document.getElementById('alphaDetailSelect');
            const alpha = parseFloat(alphaSelect.value);
            
            if (isNaN(alpha)) {{
                document.getElementById('alphaDetailInfo').innerHTML = '<p>请选择α值</p>';
                document.getElementById('alphaDistributionChart').innerHTML = '';
                return;
            }}
            
            const detailedData = data.test2.detailed[alpha.toString()];
            if (!detailedData) {{
                document.getElementById('alphaDetailInfo').innerHTML = '<p>无该α值的详细数据</p>';
                return;
            }}
            
            // 显示信息
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">α值</div>
                        <div class="info-value">${{alpha.toFixed(2)}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">候选路径数</div>
                        <div class="info-value">${{detailedData.num_candidates}}</div>
                    </div>
                </div>
                <div class="explanation" style="margin-top: 15px;">
                    <strong>图表说明：</strong>
                    下图展示了算法搜索到的所有 ${{detailedData.num_candidates}} 条候选路径的出发时间CDF分布。
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>红色粗线</strong>：算法选择的最优路径（在α=${{alpha.toFixed(2)}}分位数处最优）</li>
                        <li><strong>其他颜色细线</strong>：搜索过程中发现的候选路径</li>
                        <li><strong>橙色虚线</strong>：α=${{alpha.toFixed(2)}}分位数位置</li>
                    </ul>
                    <p style="margin-top: 10px;">
                        算法选择红色路径是因为它在橙色虚线位置的横坐标值（出发时间）最大，
                        即该路径允许最晚的出发时间，同时保证以α=${{alpha.toFixed(2)}}的可靠性按时到达。
                    </p>
                </div>
            `;
            document.getElementById('alphaDetailInfo').innerHTML = info;
            
            // 生成SVG分布对比图
            const svg = createDistributionComparisonSVG(detailedData.all_paths, alpha);
            document.getElementById('alphaDistributionChart').innerHTML = svg;
            
            // 更新地图
            if (!maps.test2 && detailedData.best_path_coords && detailedData.best_path_coords.length > 0) {{
                const center = [
                    detailedData.best_path_coords.reduce((s, c) => s + c[0], 0) / detailedData.best_path_coords.length,
                    detailedData.best_path_coords.reduce((s, c) => s + c[1], 0) / detailedData.best_path_coords.length
                ];
                maps.test2 = L.map('test2Map').setView(center, 13);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(maps.test2);
            }}
            
            if (maps.test2 && detailedData.best_path_coords && detailedData.best_path_coords.length > 0) {{
                // 清除旧图层
                maps.test2.eachLayer(layer => {{
                    if (layer instanceof L.Polyline || layer instanceof L.CircleMarker) {{
                        maps.test2.removeLayer(layer);
                    }}
                }});
                
                // 绘制路径
                const polyline = L.polyline(detailedData.best_path_coords, {{color: '#FF0000', weight: 5}}).addTo(maps.test2);
                maps.test2.fitBounds(polyline.getBounds());
                
                // 起终点
                L.circleMarker(detailedData.best_path_coords[0], {{
                    radius: 10, fillColor: '#00ff00', color: '#006600',
                    weight: 2, fillOpacity: 0.8
                }}).addTo(maps.test2).bindPopup('起点');
                
                L.circleMarker(detailedData.best_path_coords[detailedData.best_path_coords.length - 1], {{
                    radius: 10, fillColor: '#ff0000', color: '#660000',
                    weight: 2, fillOpacity: 0.8
                }}).addTo(maps.test2).bindPopup('终点');
            }}
        }}
        
        // ═══════════════════════════════════════════════════════════════════
        // SVG生成函数：分布对比图
        // ═══════════════════════════════════════════════════════════════════
        
        function createDistributionComparisonSVG(allPaths, alpha) {{
            if (! allPaths || allPaths.length === 0) return '<p>无数据</p>';
            
            const width = 1200, height = 500;
            const margin = {{top: 60, right: 50, bottom: 80, left: 80}};
            const chartWidth = width - margin.left - margin.right;
            const chartHeight = height - margin.top - margin.bottom;
            
            // 计算所有值的范围
            let allValues = [];
            allPaths.forEach(p => allValues = allValues.concat(p.values));
            const minVal = Math.min(...allValues) / 10;
            const maxVal = Math.max(...allValues) / 10;
            const valRange = maxVal - minVal;
            
            let svg = `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg" id="distributionSVG">`;
            
            // 标题
            svg += `<text x="${{width/2}}" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">`;
            svg += `候选路径出发时间分布对比 (α=${{alpha.toFixed(2)}})</text>`;
            
            // 坐标轴
            const chartX = margin.left;
            const chartY = margin.top;
            svg += `<line x1="${{chartX}}" y1="${{chartY + chartHeight}}" x2="${{chartX + chartWidth}}" y2="${{chartY + chartHeight}}" stroke="#333" stroke-width="2"/>`;
            svg += `<line x1="${{chartX}}" y1="${{chartY}}" x2="${{chartX}}" y2="${{chartY + chartHeight}}" stroke="#333" stroke-width="2"/>`;
            
            // Y轴刻度（CDF: 0-1）
            for (let i = 0; i <= 5; i++) {{
                const yVal = i / 5;
                const py = chartY + chartHeight - (i / 5) * chartHeight;
                svg += `<text x="${{chartX - 10}}" y="${{py + 5}}" text-anchor="end" font-size="11">${{yVal.toFixed(1)}}</text>`;
                svg += `<line x1="${{chartX}}" y1="${{py}}" x2="${{chartX + chartWidth}}" y2="${{py}}" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>`;
            }}
            
            // X轴刻度
            for (let i = 0; i <= 5; i++) {{
                const xVal = minVal + (i / 5) * valRange;
                const px = chartX + (i / 5) * chartWidth;
                svg += `<text x="${{px}}" y="${{chartY + chartHeight + 25}}" text-anchor="middle" font-size="11">${{xVal.toFixed(0)}}</text>`;
            }}
            
            // 绘制每条路径的CDF
            const colors = ['#4444FF', '#44FF44', '#FF44FF', '#FFAA44', '#44AAFF', '#AA44FF'];
            
            allPaths.forEach((pathInfo, idx) => {{
                const values = pathInfo.values.slice().sort((a, b) => a - b);
                const n = values.length;
                
                // 构建路径
                let pathData = 'M';
                values.forEach((val, i) => {{
                    const xNorm = (val/10 - minVal) / valRange;
                    const px = chartX + xNorm * chartWidth;
                    const py = chartY + chartHeight - ((i+1)/n) * chartHeight;
                    pathData += ` ${{px}},${{py}}`;
                }});
                
                // 样式
                const color = pathInfo.is_best ? '#FF0000' : colors[idx % colors.length];
                const strokeWidth = pathInfo.is_best ? 4 : 1.5;
                const opacity = pathInfo.is_best ?  1.0 : 0.4;
                
                svg += `<path d="${{pathData}}" fill="none" stroke="${{color}}" stroke-width="${{strokeWidth}}" opacity="${{opacity}}"/>`;
            }});
            
            // α分位数线
            const quantileLineY = chartY + chartHeight - (1-alpha) * chartHeight;
            svg += `<line x1="${{chartX}}" y1="${{quantileLineY}}" x2="${{chartX + chartWidth}}" y2="${{quantileLineY}}" `;
            svg += `stroke="orange" stroke-width="2" stroke-dasharray="8,4"/>`;
            svg += `<text x="${{chartX + chartWidth - 5}}" y="${{quantileLineY - 5}}" text-anchor="end" font-size="12" fill="orange" font-weight="bold">`;
            svg += `α=${{alpha.toFixed(2)}} 分位数</text>`;
            
            // 轴标签
            svg += `<text x="${{width/2}}" y="${{height - 10}}" text-anchor="middle" font-size="14" font-weight="bold">出发时间 (分钟)</text>`;
            svg += `<text x="20" y="${{chartY + chartHeight/2}}" text-anchor="middle" font-size="14" font-weight="bold" `;
            svg += `transform="rotate(-90 20 ${{chartY + chartHeight/2}})">累积概率 (CDF)</text>`;
            
            // 图例
            const legendX = chartX + 20;
            const legendY = chartY + 20;
            let legendHeight = 25 * Math.min(allPaths.length, 6);
            svg += `<rect x="${{legendX - 10}}" y="${{legendY - 15}}" width="200" height="${{legendHeight}}" `;
            svg += `fill="white" stroke="#ccc" stroke-width="1" opacity="0.9"/>`;
            
            let legendCount = 0;
            allPaths.forEach((pathInfo, idx) => {{
                if (legendCount >= 6) return;
                
                const color = pathInfo.is_best ? '#FF0000' : colors[idx % colors.length];
                const label = pathInfo.is_best ?  `最优路径 (长度${{pathInfo.path_length}})` : `候选${{idx+1}}`;
                
                const ly = legendY + legendCount * 25;
                svg += `<line x1="${{legendX}}" y1="${{ly}}" x2="${{legendX + 30}}" y2="${{ly}}" stroke="${{color}}" stroke-width="3"/>`;
                svg += `<text x="${{legendX + 40}}" y="${{ly + 5}}" font-size="11">${{label}}</text>`;
                
                legendCount++;
            }});
            
            svg += '</svg>';
            return svg;
        }}
        
        // ═══════════════════════════════════════════════════════════════════
        // 测试5初始化
        // ═══════════════════════════════════════════════════════════════════
        
        function initTest5() {{
            if (data.test5.length > 0) {{
                document.getElementById('odSelect').selectedIndex = 0;
                updateODView();
            }}
        }}
        
        function updateODView() {{
            const idx = parseInt(document.getElementById('odSelect').value);
            if (isNaN(idx) || ! data.test5[idx]) return;
            
            const result = data.test5[idx];
            
            // 基本信息卡片（增强版 - 包含α和目标到达时间）
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">起点</div>
                        <div class="info-value">${{result.origin}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">终点</div>
                        <div class="info-value">${{result.destination}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">可靠性要求 (α)</div>
                        <div class="info-value">${{result.alpha.toFixed(2)}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">目标到达时间</div>
                        <div class="info-value">${{result.target_arrival_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">最晚出发时间</div>
                        <div class="info-value">${{result.latest_departure_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">期望出发时间</div>
                        <div class="info-value">${{result.expected_departure_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">预留时间</div>
                        <div class="info-value">${{result.reserved_time.toFixed(1)}}分</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">路径长度</div>
                        <div class="info-value">${{result.path_length}}节点</div>
                    </div>
                </div>
            `;
            document.getElementById('test5Info').innerHTML = info;
            
            // 详细路径信息
            const pathDetails = formatPathNodes(result.path);
            document.getElementById('test5PathDetails').innerHTML = pathDetails;
            
            // 更新地图
            if (!maps.test5 && result.path_coords && result.path_coords.length > 0) {{
                const center = [
                    result.path_coords.reduce((s, c) => s + c[0], 0) / result.path_coords.length,
                    result.path_coords.reduce((s, c) => s + c[1], 0) / result.path_coords.length
                ];
                maps.test5 = L.map('test5Map').setView(center, 12);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(maps.test5);
            }}
            
            if (maps.test5 && result.path_coords && result.path_coords.length > 0) {{
                // 清除旧图层
                maps.test5.eachLayer(layer => {{
                    if (layer instanceof L.Polyline || layer instanceof L.CircleMarker) {{
                        maps.test5.removeLayer(layer);
                    }}
                }});
                
                // 绘制路径
                const polyline = L.polyline(result.path_coords, {{color: '#FF5722', weight: 5}}).addTo(maps.test5);
                maps.test5.fitBounds(polyline.getBounds());
                
                // 起终点
                L.circleMarker(result.path_coords[0], {{
                    radius: 10, fillColor: '#00ff00', color: '#006600',
                    weight: 2, fillOpacity: 0.8
                }}).addTo(maps.test5).bindPopup('起点');
                
                L.circleMarker(result.path_coords[result.path_coords.length - 1], {{
                    radius: 10, fillColor: '#ff0000', color: '#660000',
                    weight: 2, fillOpacity: 0.8
                }}).addTo(maps.test5).bindPopup('终点');
            }}
        }}
        
        // ═══════════════════════════════════════════════════════════════════
        // 路径节点格式化函数
        // ═══════════════════════════════════════════════════════════════════
        
        function formatPathNodes(path) {{
            if (!path || path.length === 0) return '<p>无路径数据</p>';
            
            const maxDisplay = 20;
            
            if (path.length <= maxDisplay) {{
                // 全部显示
                const nodesHtml = path.map(node => `<span class="path-node">${{node}}</span>`).join(' → ');
                return `
                    <div class="detail-card">
                        <h3 style="margin-bottom: 15px; color: #667eea;">🛣️ 完整路径节点</h3>
                        <div class="path-container">
                            ${{nodesHtml}}
                        </div>
                    </div>
                `;
            }} else {{
                // 折叠显示
                const visibleNodes = path.slice(0, 10);
                const hiddenNodes = path.slice(10, -5);
                const endNodes = path.slice(-5);
                
                const visibleHtml = visibleNodes.map(node => `<span class="path-node">${{node}}</span>`).join(' → ');
                const endHtml = endNodes.map(node => `<span class="path-node">${{node}}</span>`).join(' → ');
                const fullPathStr = path.join(' → ');
                
                const collapseId = `path_${{Date.now()}}_${{Math.random()}}`;
                
                return `
                    <div class="detail-card">
                        <h3 style="margin-bottom: 15px; color: #667eea;">🛣️ 路径节点 (共 ${{path.length}} 个)</h3>
                        <div class="path-container">
                            <div class="path-visible">
                                ${{visibleHtml}}
                                <button class="path-expand-btn" onclick="togglePath('${{collapseId}}')">
                                    ...(还有${{hiddenNodes.length}}个节点) ...
                                </button>
                                ${{endHtml}}
                            </div>
                            <div id="${{collapseId}}" class="path-hidden" style="display:none;">
                                <div class="path-full-details">
                                    <strong>完整路径:</strong><br>
                                    ${{fullPathStr}}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }}
        }}
        
        function togglePath(pathId) {{
            const element = document.getElementById(pathId);
            if (element) {{
                element.style.display = element.style.display === 'none' ? 'block' : 'none';
            }}
        }}
        
        // ═══════════════════════════════════════════════════════════════════
        // SVG导出功能
        // ═══════════════════════════════════════════════════════════════════
        
        function exportSVG(containerId, filename) {{
            const container = document.getElementById(containerId);
            if (!container) {{
                alert('找不到SVG容器');
                return;
            }}
            
            const svgElement = container.querySelector('svg');
            if (!svgElement) {{
                alert('没有可导出的SVG图表');
                return;
            }}
            
            // 序列化SVG
            const serializer = new XMLSerializer();
            let svgString = serializer.serializeToString(svgElement);
            
            // 添加XML声明
            svgString = '<?xml version="1.0" encoding="UTF-8"?>\\n' + svgString;
            
            // 创建Blob并下载
            const blob = new Blob([svgString], {{type: 'image/svg+xml;charset=utf-8'}});
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `${{filename}}.svg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            URL.revokeObjectURL(url);
            
            console.log('SVG已导出:', filename);
        }}
    </script>
</body>
</html>'''
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 写入文件
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n  ✓ HTML可视化文件已生成: {output_file}")
    print(f"  ✓ 包含SVG导出功能")
    print(f"  ✓ 测试2: 可视化所有候选路径分布")
    print(f"  ✓ 测试5: 完整展示 α、目标到达时间、完整路径节点")
    print(f"\n  请在浏览器中打开查看")
    print(f"{'='*70}\n")