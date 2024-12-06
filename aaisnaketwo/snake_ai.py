import numpy as np
from scipy.special import softmax
import random
from typing import Tuple, Sequence
import pygame

class SimpleModel:
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                # Initialize with much smaller weights to start with more random behavior
                self.DNA.append(np.random.randn(dim, dims[i + 1]) * 0.01)

    def update(self, obs: Sequence, temperature: float = 2.0) -> np.ndarray:
        # Increased temperature for more random initial behavior
        x = np.array(obs, dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        for i, layer in enumerate(self.DNA):
            x = x @ layer
            if i < len(self.DNA) - 1:
                x = np.tanh(x)
        
        x = x / temperature
        probs = softmax(x, axis=-1)
        return probs.flatten()

    def action(self, obs: Sequence) -> int:
        probs = self.update(obs)
        
        # Add more randomness to initial behavior
        if random.random() < 0.3:  # 30% chance of random move
            return random.randint(0, 3)
            
        obstacles = obs[4:]
        valid_moves = np.where(np.array(obstacles) == 0)[0]
        
        if len(valid_moves) > 0:
            valid_probs = probs[valid_moves]
            valid_probs = valid_probs / valid_probs.sum()
            
            # Sometimes choose random valid move instead of best move
            if random.random() < 0.2:  # 20% chance of random valid move
                return random.choice(valid_moves)
                
            valid_action_idx = np.argmax(valid_probs)
            return valid_moves[valid_action_idx]
        return np.argmax(probs)

    def mutate(self, mutation_rate: float) -> None:
        if random.random() < 0.5:  # 50% chance for random mutation
            # Completely random weights, but small
            for i, dim in enumerate(self.dims):
                if i < len(self.dims) - 1:
                    self.DNA[i] = np.random.randn(dim, self.dims[i + 1]) * 0.01
        else:  # 50% chance for small mutations
            for layer in self.DNA:
                if random.random() < mutation_rate:
                    mutation_mask = np.random.rand(*layer.shape) < 0.2  # 20% of weights
                    gaussian_noise = np.random.normal(0, 0.05, layer.shape)  # Smaller changes
                    layer += mutation_mask * gaussian_noise

    def __add__(self, other):
        baby = type(self)(dims=self.dims)
        baby_DNA = []
        for mom_layer, dad_layer in zip(self.DNA, other.DNA):
            if random.random() < 0.5:  # 50% chance to inherit from either parent
                baby_layer = mom_layer.copy()
            else:
                baby_layer = dad_layer.copy()
            baby_DNA.append(baby_layer)
        baby.DNA = baby_DNA
        return baby

# Helper Functions remain the same
def update_input_tables(snake, food, grid):
    head = snake.p
    food_pos = food.p

    FoodTable = [0, 0, 0, 0]  # N, S, E, W
    ObstacleTable = [0, 0, 0, 0]  # N, S, E, W

    if food_pos.y < head.y:
        FoodTable[0] = 1
    elif food_pos.y > head.y:
        FoodTable[1] = 1
    if food_pos.x > head.x:
        FoodTable[2] = 1
    elif food_pos.x < head.x:
        FoodTable[3] = 1

    directions = {
        0: Vector(head.x, head.y - 1),
        1: Vector(head.x, head.y + 1),
        2: Vector(head.x + 1, head.y),
        3: Vector(head.x - 1, head.y),
    }
    
    for i, direction in directions.items():
        if not direction.within(grid) or direction in snake.body:
            ObstacleTable[i] = 1

    return FoodTable + ObstacleTable

def fitness_function(steps, food_count, max_steps=500):
    if food_count == 0:
        # Harsher death penalty
        base_penalty = -500
        # Additional penalty based on how quickly it died
        quick_death_penalty = -((max_steps - steps) / max_steps) * 500
        return base_penalty + quick_death_penalty
    
    # Smaller base reward for food
    food_reward = food_count * 500
    
    # Smaller efficiency bonus
    efficiency_bonus = max(0, (max_steps - steps) / food_count) * 5
    
    # Harsher step penalty
    step_penalty = -steps
    
    return food_reward + efficiency_bonus + step_penalty

def simulate_game(agent, game, max_steps=500):
    snake = Snake(game=game)
    food = Food(game=game)
    steps = 0
    food_count = 0
    
    # Add early termination for snakes that move in circles
    last_positions = []
    repeated_positions = 0

    while steps < max_steps:
        obs = update_input_tables(snake, food, game.grid)
        action = agent.action(obs)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()

        # Track position history
        current_pos = (snake.p.x, snake.p.y)
        last_positions.append(current_pos)
        if len(last_positions) > 20:  # Check last 20 moves
            last_positions.pop(0)
            # Count repeated positions
            if current_pos in last_positions[:-1]:
                repeated_positions += 1
                if repeated_positions > 10:  # Break if too many repeated positions
                    break

        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break

        if snake.p == food.p:
            snake.add_score()
            food = Food(game=game)
            food_count += 1
            # Reset repeated positions counter when food is eaten
            repeated_positions = 0

        steps += 1

    return steps, food_count

def train_agents(agents, generations, mutation_rate):
    for generation in range(generations):
        fitness_scores = []
        food_counts = []
        
        for agent in agents:
            game = SnakeGame()
            steps, food_count = simulate_game(agent, game)
            fitness_scores.append(fitness_function(steps, food_count))
            food_counts.append(food_count)
            del game

        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        best_food = max(food_counts)
        avg_food = sum(food_counts) / len(food_counts)
        print(f'Generation {generation + 1}:')
        print(f'Best fitness = {best_fitness:.2f}, Avg fitness = {avg_fitness:.2f}')
        print(f'Best food = {best_food}, Avg food = {avg_food:.2f}')

        # Sort agents by fitness
        sorted_agents = [agent for _, agent in sorted(zip(fitness_scores, agents), 
                                                    key=lambda pair: pair[0], reverse=True)]
        
        # Keep top 25% unchanged
        elite_count = len(agents) // 4
        new_agents = sorted_agents[:elite_count]
        
        # Generate rest of population
        while len(new_agents) < len(agents):
            if random.random() < 0.3:  # 30% chance for completely new random agent
                new_agents.append(SimpleModel(dims=(8, 12, 4)))
            else:  # 70% chance for breeding
                parent1, parent2 = random.sample(sorted_agents[:len(sorted_agents)//2], 2)
                child = parent1 + parent2
                child.mutate(mutation_rate)
                new_agents.append(child)

        agents = new_agents

    return agents[0]

def show_game_with_agent(agent):
    game = SnakeGame()
    snake = Snake(game=game)
    food = Food(game=game)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        obs = update_input_tables(snake, food, game.grid)
        action = agent.action(obs)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()

        if not snake.p.within(game.grid) or snake.cross_own_tail:
            print(f'Final Score: {snake.score}')
            running = False

        if snake.p == food.p:
            snake.add_score()
            food = Food(game=game)

        game.screen.fill((0, 0, 0))
        pygame.draw.rect(game.screen, game.color_food, game.block(food.p))
        for segment in snake.body:
            pygame.draw.rect(game.screen, game.color_snake_head, game.block(segment))

        pygame.display.update()
        game.clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    from snake import Vector, Snake, Food, SnakeGame
    
    population_size = 50
    generations = 50
    mutation_rate = 0.1

    agents = [SimpleModel(dims=(8, 12, 4)) for _ in range(population_size)]
    best_agent = train_agents(agents, generations, mutation_rate)
    show_game_with_agent(best_agent)