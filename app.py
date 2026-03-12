import os
from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """HW1-2: 隨機策略生成 + 策略評估 (Policy Evaluation)
    
    使用隨機策略 (Stochastic Policy)：每個動作機率 = 1/4
    公式：V(s) = Σ π(a|s) × [R(s,a) + γ·V(s')]
              = 0.25 × [R+γV(上)] + 0.25 × [R+γV(下)] + 0.25 × [R+γV(左)] + 0.25 × [R+γV(右)]
    """
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = [tuple(obs) for obs in data['obstacles']]

    actions = ['U', 'D', 'L', 'R']
    prob = 1.0 / len(actions)  # 均勻隨機策略：每個動作 0.25

    # 策略評估 (Policy Evaluation) — Bellman Equation with stochastic policy
    V = {(r, c): 0.0 for r in range(n) for c in range(n)}

    gamma = 0.9
    reward_step = -1
    reward_goal = 10
    theta = 1e-6
    max_iterations = 1000

    for iteration in range(max_iterations):
        delta = 0
        V_new = V.copy()
        for r in range(n):
            for c in range(n):
                state = (r, c)
                if state == end or state in obstacles:
                    continue

                # 動作機率加權和：Σ π(a|s) × [R + γ·V(s')]
                weighted_sum = 0.0
                for a in actions:
                    nr, nc = get_next(r, c, a)
                    next_state = (nr, nc)

                    # 邊界 / 障礙物檢查：撞牆則留在原地
                    if nr < 0 or nr >= n or nc < 0 or nc >= n or next_state in obstacles:
                        next_state = state

                    cur_reward = reward_goal if next_state == end else reward_step
                    weighted_sum += prob * (cur_reward + gamma * V[next_state])

                v_old = V[state]
                V_new[state] = weighted_sum
                delta = max(delta, abs(v_old - V_new[state]))

        V = V_new
        if delta < theta:
            break

    # 隨機策略用於顯示（每個格子隨機一個代表方向）
    policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) == end or (r, c) in obstacles:
                policy[(r, c)] = None
            else:
                policy[(r, c)] = random.choice(actions)

    # 格式化輸出
    v_matrix, p_matrix = format_matrices(n, V, policy, start, end, obstacles)

    return jsonify({
        'v_matrix': v_matrix,
        'p_matrix': p_matrix,
        'iterations': iteration + 1,
        'gamma': gamma
    })


@app.route('/value_iteration', methods=['POST'])
def value_iteration():
    """HW1-3: 價值迭代 (Value Iteration) + 策略萃取 (Policy Extraction)"""
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = [tuple(obs) for obs in data['obstacles']]

    actions = ['U', 'D', 'L', 'R']

    # 價值迭代 (Value Iteration) — 使用 Max 運算
    V = {(r, c): 0.0 for r in range(n) for c in range(n)}

    gamma = 0.9
    reward_step = -1
    reward_goal = 10
    theta = 1e-6
    max_iterations = 1000

    for iteration in range(max_iterations):
        delta = 0
        V_new = V.copy()
        for r in range(n):
            for c in range(n):
                state = (r, c)
                if state == end or state in obstacles:
                    continue

                # 對所有動作取 Max（與 PE 的核心差異）
                max_val = float('-inf')
                for a in actions:
                    nr, nc = get_next(r, c, a)
                    next_state = (nr, nc)

                    if nr < 0 or nr >= n or nc < 0 or nc >= n or next_state in obstacles:
                        next_state = state

                    cur_reward = reward_goal if next_state == end else reward_step
                    val = cur_reward + gamma * V[next_state]

                    if val > max_val:
                        max_val = val

                v_old = V[state]
                V_new[state] = max_val
                delta = max(delta, abs(v_old - V_new[state]))

        V = V_new
        if delta < theta:
            break

    # 策略萃取 (Policy Extraction) — 使用 Argmax 運算
    policy = {}
    for r in range(n):
        for c in range(n):
            state = (r, c)
            if state == end or state in obstacles:
                policy[state] = None
                continue

            best_action = None
            best_val = float('-inf')
            for a in actions:
                nr, nc = get_next(r, c, a)
                next_state = (nr, nc)

                if nr < 0 or nr >= n or nc < 0 or nc >= n or next_state in obstacles:
                    next_state = state

                cur_reward = reward_goal if next_state == end else reward_step
                val = cur_reward + gamma * V[next_state]

                if val > best_val:
                    best_val = val
                    best_action = a

            policy[state] = best_action

    # 最佳路徑萃取
    path = extract_path(start, end, policy, obstacles, n)

    # 格式化輸出
    v_matrix, p_matrix = format_matrices(n, V, policy, start, end, obstacles)

    return jsonify({
        'v_matrix': v_matrix,
        'p_matrix': p_matrix,
        'iterations': iteration + 1,
        'gamma': gamma,
        'path': path
    })


def get_next(r, c, action):
    """根據動作取得下一個座標"""
    if action == 'U': return r - 1, c
    elif action == 'D': return r + 1, c
    elif action == 'L': return r, c - 1
    elif action == 'R': return r, c + 1
    return r, c


def extract_path(start, end, policy, obstacles, n):
    """從起點沿著策略走到終點，萃取最佳路徑"""
    path = [list(start)]
    current = start
    visited = set()
    max_steps = n * n

    while current != end and len(path) < max_steps:
        if current in visited:
            break
        visited.add(current)

        a = policy.get(current)
        if a is None:
            break

        nr, nc = get_next(current[0], current[1], a)
        next_state = (nr, nc)

        if nr < 0 or nr >= n or nc < 0 or nc >= n or next_state in obstacles:
            break

        current = next_state
        path.append(list(current))

    return path


def format_matrices(n, V, policy, start, end, obstacles):
    """格式化 Value Matrix 和 Policy Matrix"""
    v_matrix = []
    p_matrix = []
    for r in range(n):
        v_row = []
        p_row = []
        for c in range(n):
            if (r, c) in obstacles:
                v_row.append('OBSTACLE')
                p_row.append('OBSTACLE')
            elif (r, c) == end:
                v_row.append(0.0)
                p_row.append('END')
            elif (r, c) == start:
                v_row.append(round(V[(r, c)], 2))
                p_row.append('START_' + (policy[(r, c)] or ''))
            else:
                v_row.append(round(V[(r, c)], 2))
                p_row.append(policy[(r, c)])
        v_matrix.append(v_row)
        p_matrix.append(p_row)

    return v_matrix, p_matrix


if __name__ == '__main__':
    app.run(debug=True, port=5000)
