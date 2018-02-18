import pygame
import pygame.camera
import tkinter
from tkinter import filedialog
import numpy
import cv2

WHITE = (255, 255, 255)
BLACK = (0,0,0)

def startGui(screen, clock):
    x1=100;y1=200;x2=100;y2=300
    w1=300;h1=80;w2=300;h2=80
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (x1, y1, w1, h1), 5)
    pygame.draw.rect(screen, BLACK, (x2, y2, w2, h2), 5)
    font = pygame.font.SysFont('Calibri', 25, True, False)
    chooseFile=font.render("choose a picture", True, BLACK)
    useCamera=font.render("take a picture", True, BLACK)
    screen.blit(chooseFile,[120,210])
    screen.blit(useCamera, [120, 310])
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pos()[0] >= x1 and pygame.mouse.get_pos()[1] >= y1:
                    if pygame.mouse.get_pos()[0] <= x1+w1 and pygame.mouse.get_pos()[1] <= y1+h1:
                        return 1
                if pygame.mouse.get_pos()[0] >= x2 and pygame.mouse.get_pos()[1] >= y2:
                    if pygame.mouse.get_pos()[0] <= x2+w2 and pygame.mouse.get_pos()[1] <= y2+h2:
                        return 2
        clock.tick(30)

def choosePicture():
    tkinter.Tk().withdraw()
    fileName = filedialog.askopenfilename(initialdir="//", title="Select file",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return fileName

def getWebcam(screen,clock,WIDTH,HEIGHT):
    camera = cv2.VideoCapture(0)
    done = False
    while not done:
        ret, frame = camera.read()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return 0;
            elif event.type == pygame.KEYDOWN:
                img_name = "C:\\Users\\10266\\Desktop\\blueprint\\PROJECT\\user.jpg".format(0)
                cv2.imwrite(img_name, frame)
                return "C:\\Users\\10266\\Desktop\\blueprint\\PROJECT\\user.jpg";
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = numpy.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (WIDTH,HEIGHT))
        screen.blit(frame, (0, 0))
        pygame.draw.rect(screen, WHITE, (0, 0, 120, 20), 0)
        font = pygame.font.SysFont('Calibri', 20, True, False)
        tips = font.render("press any key", True, BLACK)
        screen.blit(tips, [0,0])
        pygame.display.update()
        clock.tick(30)

def countDown(screen, clock):
    for i in range(0,90):
        screen.fill(WHITE)
        font = pygame.font.SysFont('Calibri', 100, True, False)
        time = font.render(str(3-i//30), True, BLACK)
        screen.blit(time, [200,50])
        word=""
        if i<45: word="We think"
        else: word="you look like"
        font = pygame.font.SysFont('Calibri', 30, True, False)
        word = font.render(word, True, BLACK)
        screen.blit(word, [100,150])
        pygame.display.update()
        clock.tick(30)

def resultDisplay(screen, loc1, loc2, person):
    font = pygame.font.SysFont('Calibri', 25, True, False)
    pygame.display.set_caption("GUI")

    pic1 = pygame.image.load(loc1)
    pic1 = pygame.transform.scale(pic1, (200, 300))
    pic2 = pygame.image.load(loc2)
    pic2 = pygame.transform.scale(pic2, (200, 300))
    person = font.render(person, True, BLACK)
    you = font.render("YOU", True, BLACK)

    screen.fill(WHITE)
    screen.blit(you, [20, 320])
    screen.blit(person, [260, 320])
    screen.blit(pic1, [20, 20])
    screen.blit(pic2, [260, 20])
    pygame.display.flip()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
    pygame.quit()