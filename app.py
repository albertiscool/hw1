import os
from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = [tuple(obs) for obs in data['obstacles']]
    
    # 隨機生成策略 (Random Policy)
    actions = ['U', 'D', 'L', 'R']
    policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) == end or (r, c) in obstacles:
                policy[(r, c)] = None
            else:
                policy[(r, c)] = random.choice(actions)
                
    # 策略評估 (Policy Evaluation)
    V = { (r, c): 0.0 for r in range(n) for c in range(n) }
    
    gamma = 0.9       # 折扣因子
    reward_step = -1   # 每步的獎勵
    reward_goal = 10   # 到達終點的獎勵
    theta = 1e-6       # 收斂閾值
    
    max_iterations = 1000
    
    for iteration in range(max_iterations):
        delta = 0
        V_new = V.copy()
        for r in range(n):
            for c in range(n):
                state = (r, c)
                # 終點和障礙物不更新
                if state == end or state in obstacles:
                    continue
                
                a = policy[state]
                next_r, next_c = r, c
                if a == 'U': next_r -= 1
                elif a == 'D': next_r += 1
                elif a == 'L': next_c -= 1
                elif a == 'R': next_c += 1
                
                next_state = (next_r, next_c)
                
                # 檢查邊界和障礙物：如果超出邊界或撞到障礙物，則留在原處
                if next_r < 0 or next_r >= n or next_c < 0 or next_c >= n or next_state in obstacles:
                    next_state = state
                    
                # 計算獎勵
                cur_reward = reward_goal if next_state == end else reward_step
                
                v_old = V[state]
                V_new[state] = cur_reward + gamma * V[next_state]
                delta = max(delta, abs(v_old - V_new[state]))
                
        V = V_new
        if delta < theta:
            break
            
    # 格式化矩陣資料給前端
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
        
    return jsonify({
        'v_matrix': v_matrix,
        'p_matrix': p_matrix,
        'iterations': iteration + 1,
        'gamma': gamma
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
