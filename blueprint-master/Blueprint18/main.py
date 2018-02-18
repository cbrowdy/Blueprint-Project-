import GUI
import comparer
import pygame
def main():
    #vars
    WIDTH = 500;HEIGHT = 420
    size = (WIDTH, HEIGHT)
    #program starts
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)

    RV=GUI.startGui(screen, clock)
    pic=""
    if RV == 0: pygame.quit();return
    elif RV == 2:
        pic = GUI.getWebcam(screen, clock, WIDTH, HEIGHT)
        if(pic==0): pygame.quit();return
    elif RV== 1:
        pic = GUI.choosePicture()
    results = comparer.getMatch(pic)
    GUI.countDown(screen, clock)

    loc1=pic
    loc2=results[0]
    person=results[1]
    GUI.resultDisplay(screen, loc1, loc2, person)

main()