import random


class GridWorld:
    """坐标为 (行, 列)，从 0 开始。"""

    ACTIONS = {
        "上": (-1, 0),
        "下": (1, 0),
        "左": (0, -1),
        "右": (0, 1),
    }

    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = {(1, 1), (2, 3)}

        self.states = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in self.obstacles
        ]
        self.actions = list(self.ACTIONS)
        self.reset()

    def reset(self):
        """重开一个回合，返回起点。"""
        self.position = self.start
        self.done = False
        return self.position

    def transitions(self, state, action):
        """
        查询环境模型，不改变智能体的实际位置。

        返回列表，每项是：
        (转移概率, 下一状态, 奖励, 是否终止)

        当前环境是确定性的，因此每个动作只有一个结果。
        """
        if state not in self.states:
            raise ValueError(f"无效状态：{state}")
        if action not in self.ACTIONS:
            raise ValueError(f"无效动作：{action}")

        # 终点没有后续奖励，价值保持为 0。
        if state == self.goal:
            return [(1.0, self.goal, 0, True)]

        row, col = state
        delta_row, delta_col = self.ACTIONS[action]
        candidate = (row + delta_row, col + delta_col)

        inside = (
            0 <= candidate[0] < self.size
            and 0 <= candidate[1] < self.size
        )

        if inside and candidate not in self.obstacles:
            next_state = candidate
        else:
            # 撞墙或障碍物，留在原地。
            next_state = state

        done = next_state == self.goal
        reward = 10 if done else -1

        return [(1.0, next_state, reward, done)]

    def step(self, action):
        """实际执行动作，返回下一位置、奖励、是否结束。"""
        if self.done:
            raise RuntimeError("回合已结束，请先调用 reset()")

        # 确定性环境，直接取唯一的转移结果。
        _, next_state, reward, done = self.transitions(
            self.position, action
        )[0]
        self.position = next_state
        self.done = done

        return next_state, reward, done


def action_value(env, state, action, values, gamma):
    """
    根据当前价值表计算一个动作的价值：

    Q(s,a) = Σ p * [r + gamma * V(s')]

    如果转移已经终止，后续价值为 0。
    """
    result = 0.0

    for probability, next_state, reward, done in env.transitions(
        state, action
    ):
        future_value = 0.0 if done else values[next_state]
        result += probability * (reward + gamma * future_value)

    return result


def greedy_action(env, state, values, gamma):
    """选择动作价值最大的动作；平局时按 actions 的顺序选择。"""
    return max(
        env.actions,
        key=lambda action: action_value(
            env, state, action, values, gamma
        ),
    )


def policy_evaluation(
    env, policy, gamma=0.9, tolerance=1e-6, max_iterations=10000
):
    """
    策略评估：计算给定策略的价值。

    policy[state][action] 表示 π(a|s)，即动作选择概率。
    """
    values = {state: 0.0 for state in env.states}

    for iteration in range(1, max_iterations + 1):
        # 同步更新：这一轮全部使用上一轮的价值。
        new_values = values.copy()
        delta = 0.0

        for state in env.states:
            if state == env.goal:
                continue

            # 贝尔曼期望更新：对动作按策略概率加权求和。
            new_values[state] = sum(
                policy[state][action]
                * action_value(env, state, action, values, gamma)
                for action in env.actions
            )

            delta = max(
                delta, abs(new_values[state] - values[state])
            )

        values = new_values

        if delta < tolerance:
            return values, iteration

    raise RuntimeError("策略评估未收敛，请检查参数")


def value_iteration(
    env, gamma=0.9, tolerance=1e-6, max_iterations=10000
):
    """值迭代：反复进行贝尔曼最优更新。"""
    values = {state: 0.0 for state in env.states}

    for iteration in range(1, max_iterations + 1):
        new_values = values.copy()
        delta = 0.0

        for state in env.states:
            if state == env.goal:
                continue

            # 值迭代核心：在所有动作价值中取最大值。
            new_values[state] = max(
                action_value(env, state, action, values, gamma)
                for action in env.actions
            )

            delta = max(
                delta, abs(new_values[state] - values[state])
            )

        values = new_values

        if delta < tolerance:
            # 从最终价值表提取贪心策略。
            policy = {
                state: greedy_action(env, state, values, gamma)
                for state in env.states
                if state != env.goal
            }
            return values, policy, iteration

    raise RuntimeError("值迭代未收敛，请检查参数")


def policy_iteration(
    env, gamma=0.9, tolerance=1e-6, max_iterations=1000
):
    """策略迭代：策略评估 → 策略改善，直到策略不变。"""

    # 初始确定性策略：所有位置都选择“上”。
    policy = {
        state: env.actions[0]
        for state in env.states
        if state != env.goal
    }
    total_evaluation_iterations = 0

    for iteration in range(1, max_iterations + 1):
        # 1. 将确定性策略转为动作概率分布。
        probability_policy = {
            state: {
                action: float(action == policy[state])
                for action in env.actions
            }
            for state in policy
        }

        # 2. 策略评估。
        values, evaluation_iterations = policy_evaluation(
            env, probability_policy, gamma, tolerance
        )
        total_evaluation_iterations += evaluation_iterations

        # 3. 策略改善：根据价值表重新选择动作。
        stable = True

        for state in policy:
            old_action = policy[state]
            new_action = greedy_action(env, state, values, gamma)

            policy[state] = new_action

            if new_action != old_action:
                stable = False

        # 4. 策略不变，算法结束。
        if stable:
            return (
                values,
                policy,
                iteration,
                total_evaluation_iterations,
            )

    raise RuntimeError("策略迭代未收敛，请检查参数")


def print_values(env, values, title):
    """打印价值表。"""
    print(f"\n{title}")

    for row in range(env.size):
        cells = []

        for col in range(env.size):
            state = (row, col)

            if state in env.obstacles:
                cells.append("    ####")
            else:
                cells.append(f"{values[state]:8.3f}")

        print(" ".join(cells))


def print_policy(env, policy, title):
    """用文字打印策略，G 为终点，# 为障碍物。"""
    print(f"\n{title}")

    for row in range(env.size):
        cells = []

        for col in range(env.size):
            state = (row, col)

            if state in env.obstacles:
                cells.append("#")
            elif state == env.goal:
                cells.append("G")
            else:
                cells.append(policy[state])

        print("  ".join(cells))


def run_episode(
    env, policy=None, rng=None, max_steps=1000, verbose=True
):
    """
    policy=None 时随机行动；
    否则按给定的确定性策略行动。
    """
    if rng is None:
        rng = random.Random()

    state = env.reset()
    total_reward = 0
    path = [state]

    for step_number in range(1, max_steps + 1):
        action = (
            rng.choice(env.actions)
            if policy is None
            else policy[state]
        )

        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state)

        if verbose:
            print(
                f"第 {step_number} 步 | "
                f"当前位置：{state} | "
                f"动作：{action} | "
                f"下一位置：{next_state} | "
                f"奖励：{reward:+d} | "
                f"是否结束：{done}"
            )

        state = next_state

        if done:
            return step_number, total_reward, True, path

    # 步数上限只是截断，不表示到达终点。
    return max_steps, total_reward, False, path


def main():
    env = GridWorld()
    gamma = 0.9
    rng = random.Random(42)

    # 一、随机智能体运行 3 个回合。
    print("========== 随机策略：运行 3 个回合 ==========")

    for episode in range(1, 4):
        print(f"\n第 {episode} 回合")

        steps, reward, done, _ = run_episode(env, rng=rng)

        print(
            f"回合结果：步数={steps}，"
            f"累计奖励={reward}，到达终点={done}"
        )
        if not done:
            print("达到步数上限，本回合截断。")

    # 二、随机策略的价值评估。
    random_policy = {
        state: {
            action: 1.0 / len(env.actions)
            for action in env.actions
        }
        for state in env.states
        if state != env.goal
    }

    random_values, evaluation_rounds = policy_evaluation(
        env, random_policy, gamma
    )
    print_values(env, random_values, "随机策略的价值表")
    print(f"策略评估更新轮数：{evaluation_rounds}")

    # 三、值迭代。
    vi_values, vi_policy, vi_rounds = value_iteration(env, gamma)

    print_values(env, vi_values, "值迭代：最优价值表")
    print_policy(env, vi_policy, "值迭代：最优策略")
    print(f"值迭代更新轮数：{vi_rounds}")

    # 四、策略迭代。
    pi_values, pi_policy, pi_rounds, pi_evaluation_rounds = (
        policy_iteration(env, gamma)
    )

    print_values(env, pi_values, "策略迭代：最优价值表")
    print_policy(env, pi_policy, "策略迭代：最优策略")
    print(f"策略评估与改善循环次数：{pi_rounds}")
    print(f"策略评估总更新轮数：{pi_evaluation_rounds}")

    # 五、比较两种算法的价值。
    max_difference = max(
        abs(vi_values[state] - pi_values[state])
        for state in env.states
    )
    print(f"\n两种算法价值表的最大差异：{max_difference:.8f}")

    # 六、实际执行求出的策略。
    for name, policy in [
        ("值迭代", vi_policy),
        ("策略迭代", pi_policy),
    ]:
        print(f"\n========== 执行{name}策略 ==========")

        steps, reward, done, path = run_episode(
            env, policy=policy
        )

        print("路径：", " → ".join(map(str, path)))
        print(
            f"结果：步数={steps}，"
            f"累计奖励={reward}，到达终点={done}"
        )


if __name__ == "__main__":
    main()