# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: quent
"""

import numpy as np
import pymunk
import pymunk.pygame_util
import pygame

import Settings
import Graphics.GraphicEngine as GE
import Player.PlayerActions as PActions
import Engine.Actions as Actions


should_continue = True

def initGame(score=np.zeros(2, dtype=np.uint8)) -> dict:
    """
    Renvoie le dictionnaire "game" initialisé pour une nouvelle partie et démarre pygame.

    Returns
    -------
    dict
        Dictionnaire "game" contenant les informations de l'espace, la balle, les joueurs etc.

    """
    
    space = pymunk.Space()
    space.damping = Settings.GROUND_FRICTION
    game = {
        "space": space,
        "score": score
        }
    
    GE.buildBoard(game)
    GE.buildBall(game)
    GE.buildPlayers(game)
    
    return game

def stopGame() -> None:
    """
    Arrête le jeu.

    Returns
    -------
    None.

    """
    
    global should_continue
    should_continue = False
    pygame.display.quit()
    pygame.quit()
    return


def main() -> None:
    """
    Boucle de jeu principale.

    Returns
    -------
    None.

    """
    
    pygame.init()
    game = initGame()
    screen, draw_options = GE.initScreen()
    
    clock = pygame.time.Clock()
    
    delta_time = Settings.DELTA_TIME
    fps = int(1000/delta_time)
    
    min_delta_time = 1000/Settings.MAX_FPS
    
    time = 0
    GE.display(game, screen, draw_options)
    
    import Engine.Vision as Vision
    print(Vision.rayTracing(game, game["selected_player"]))
    return
        
    while(should_continue):
        
        PActions.process_events(game, stopGame)
        if(should_continue):
            # Décision de l'ia ?
            
            for step in range(delta_time):
                game["space"].step(0.001)
            time += delta_time
            Actions.reset_movements(game)
            
            if(time >= min_delta_time):
                GE.display(game, screen, draw_options)
                time -= min_delta_time
            clock.tick(fps)
            checkIfGoal(game)
            #TODO: vérifier si un joueur est sorti
    return


def checkIfGoal(game) -> np.uint8:
    """
    Vérifie si un but a été marqué, et par qui. Rstart une manche en conséquence.

    Parameters
    ----------
    game : TYPE
        DESCRIPTION.

    Returns
    -------
    np.uint8:
        0: pas de but
        1: gauche à marqué
        2: droite à marqué

    """
    
    body, shape = game["ball"]
    ball_x = body.position[0]
    dim_x = Settings.DIM_X
    offset = Settings.SCREEN_OFFSET
    
    if(ball_x < offset):
        # Goal à gauche
        game["score"][1] += 1
        
        new_game = initGame(game["score"])
        game.update(new_game)
        
        print("Les joueurs à droite ont marqué !")
        return np.uint8(2)
    
    elif(ball_x > dim_x+offset):
        # Goal à droite
        game["score"][0] += 1
        
        new_game = initGame(game["score"])
        game.update(new_game)
        
        print("Les joueurs à gauche ont marqué !")
        return np.uint8(1)
    
    else:
        return np.uint8(0)


main()































