import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

# Define the neural network
model = keras.Sequential([
    keras.layers.Dense(128, input_dim=5, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Initialize game variables
ball_x, ball_y = 300, 200
ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
paddle1_y, paddle2_y = 150, 150
score1, score2 = 0, 0

# Preprocess the state
def preprocess_state(state):
    return np.array(state).reshape(1, -1)

# Update game state
def update_game_state(action):
    global ball_x, ball_y, ball_dx, ball_dy, paddle1_y, paddle2_y, score1, score2
    if action == 0 and paddle1_y > 0:
        paddle1_y -= 30
    elif action == 2 and paddle1_y < 300:
        paddle1_y += 30
    # Let the other paddle (paddle2_y) move towards the ball with some randomness
    if ball_dy > 0 and ball_x > 300:
        if paddle2_y + 50 < ball_y and paddle2_y < 300 and random.random() < 0.8:
            paddle2_y += 30
        elif paddle2_y + 50 > ball_y and paddle2_y > 0 and random.random() < 0.8:
            paddle2_y -= 30
        else:
            # add randomness for missing the ball occasionally
            if random.random() < 0.01:
                if paddle2_y < 200:
                    paddle2_y += 30
                else:
                    paddle2_y -= 30
    # Update the position of the ball
    ball_x += ball_dx
    ball_y += ball_dy
    if ball_y < 10 or ball_y > 390:
        ball_dy *= -1
    if ball_x < 30 and paddle1_y + 100 >= ball_y >= paddle1_y - 10:
        ball_dx *= -1
        ball_x = 30 + 10
    elif ball_x > 570 and paddle2_y + 100 >= ball_y >= paddle2_y - 10:
        ball_dx *= -1
        ball_x = 570 - 10
    elif ball_x < 10 or ball_x > 590:
        ball_x, ball_y = 300, 200
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        score1, score2 = 0, 0

# Draw game objects
def draw_game_objects():
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, paddle1_y, 10, 100))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(590, paddle2_y, 10, 100))
    pygame.draw.circle(screen, (255, 255, 255), (int(ball_x), int(ball_y)), 10)
    pygame.draw.line(screen, (255, 255, 255), (300, 0), (300, 400))
    font = pygame.font.SysFont(None, 30)
    score_text = font.render(str(score1) + " - " + str(score2), True, (255, 255, 255))
    screen.blit(score_text, (260, 10))
    pygame.display.flip()

# Run game loop
def run_game_loop():
  global ball_x, ball_y, ball_dx, ball_dy, paddle1_y, paddle2_y, score1, score2
  state = [ball_x, ball_y, ball_dx, ball_dy, paddle1_y]
  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
    # Get an action from the model
    state = preprocess_state(state)
    action = np.argmax(model.predict(state)[0])
    # Update game state based on the action
    update_game_state(action)
    # Draw game objects
    draw_game_objects()
    # Preprocess the next state
    next_state = [ball_x, ball_y, ball_dx, ball_dy, paddle1_y]
    # Get the reward
    if score1 > score2:
      reward = 1
    elif score1 < score2:
      reward = -1
    else:
      reward = 0
    # Check if the game is over
    done = score1 >= 10 or score2 >= 10
    if done:
      return
    next_state = preprocess_state(next_state)
    target = model.predict(state)
    target[0][action] = reward + 0.99 * np.amax(model.predict(next_state)[0])
    model.fit(state, target, epochs=1, verbose=0)
    state = next_state

# train the model
for i in range(1000):
  run_game_loop()

# save the model
model.save('pong_model.h5')

# quit pygame
pygame.quit()
