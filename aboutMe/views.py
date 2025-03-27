from django.shortcuts import render
from django.conf import settings
from django.urls import reverse
import os


def base(request):
    intor_en_file_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/self-intro_en.txt')
    skills_en_file_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/skills_en.txt')
    try:
        with open(intor_en_file_path, 'r', encoding='utf-8') as intro_file:
            intro_en = intro_file.read()
            intro_en = intro_en.replace('\n', '<br>')
            intro_en = intro_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
            project_url = reverse('project')
            experience_url = reverse('experience')
            intro_en = intro_en.replace('_Project_', f'<a href="{project_url}" class="hyperText">Project</a>')
            intro_en = intro_en.replace('_Experience_', f'<a href="{experience_url}" class="hyperText">Experience</a>')
    except FileNotFoundError:
        intro_en = "error: file not found"
    try:
        with open(skills_en_file_path, 'r', encoding='utf-8') as skills_file:
            skills_en = skills_file.read()
            skills_en = skills_en.replace('\n', '<br>')
    except FileNotFoundError:
        skills_en = "error: file not found"
    return render(request, 'base.html', {'intro_en': intro_en, 'skills_en': skills_en})

def blog(request):
    return render(request, 'blog.html')

def contact(request):
    return render(request, 'contact.html')

def experience(request):
    return render(request, 'experience.html')

def project(request):
    return render(request, 'project.html')

def vex(request):
    vex_en_file_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/vex-experience_en.txt')
    try:
        with open(vex_en_file_path, 'r', encoding='utf-8') as vex_file:
            vex_en = vex_file.read()
            vex_en = vex_en.replace('\n', '<br>')
            vex_en = vex_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        vex_en = "error: file not found"
    return render(request, 'vex.html', {'vex_en': vex_en})

def discordMusicPlayerBot(request):
    dcBotIntro_en_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/dcBotIntro_en.txt')
    dcBot_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/dc_bot.py')
    try:
        with open(dcBot_path, 'r', encoding='utf-8') as dcBot_file:
            dcBotCode = dcBot_file.read()
    except FileNotFoundError:
        dcBotCode = "error: file not found"
    try:
        with open(dcBotIntro_en_path, 'r', encoding='utf-8') as dcBotIntro_en_file:
            dcBotIntro_en = dcBotIntro_en_file.read()
            dcBotIntro_en = dcBotIntro_en.replace('\n', '<br>')
            dcBotIntro_en = dcBotIntro_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        dcBotIntro_en = "error: file not found"
    return render(request, 'discordMusicPlayerBot.html', {'dcBotCode': dcBotCode, 'dcBotIntro_en': dcBotIntro_en,})

def othello(request):
    othelloIntro_en_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/othelloIntro_en.txt')
    othello_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/othello.py')
    othelloRR_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/random_vs_random_6x6.txt')
    othelloHR_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/human_vs_random_4x4.txt')
    try:
        with open(othelloIntro_en_path, 'r', encoding='utf-8') as othelloIntro_en_file:
            othelloIntro_en = othelloIntro_en_file.read()
            othelloIntro_en = othelloIntro_en.replace('\n', '<br>')
            othelloIntro_en = othelloIntro_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        othelloIntro_en = "error: file not found"
    try:
        with open(othello_path, 'r', encoding='utf-8') as othello_file:
            othello = othello_file.read()
    except FileNotFoundError:
        othello = "error: file not found"
    try:
        with open(othelloRR_path, 'r', encoding='utf-8') as othelloRR_file:
            othelloRR = othelloRR_file.read()
    except FileNotFoundError:
        othelloRR = "error: file not found"
    try:
        with open(othelloHR_path, 'r', encoding='utf-8') as othelloHR_file:
            othelloHR = othelloHR_file.read()
    except FileNotFoundError:
        othelloHR = "error: file not found"
    return render(request, 'othello.html', {'othello': othello, 'othelloRR': othelloRR, 'othelloHR': othelloHR,'othelloIntro_en': othelloIntro_en,})

def sokoban(request):
    sokobanIntro_en_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/sokobanIntro_en.txt')
    socoban_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/@推箱子@.py')
    try:
        with open(sokobanIntro_en_path, 'r', encoding='utf-8') as sokobanIntro_en_file:
            sokobanIntro_en = sokobanIntro_en_file.read()
            sokobanIntro_en = sokobanIntro_en.replace('\n', '<br>')
            sokobanIntro_en = sokobanIntro_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        sokobanIntro_en = "error: file not found"
    try:
        with open(socoban_path, 'r', encoding='utf-8') as socoban_file:
            sokobanCode = socoban_file.read()
    except FileNotFoundError:
        sokobanCode = "error: file not found"
    return render(request, 'sokoban.html', {'sokobanCode': sokobanCode, 'sokobanIntro_en': sokobanIntro_en,})

def thisWebPage(request):
    this_web_en_file_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/this-web_en.txt')
    views_path = os.path.join(settings.BASE_DIR, 'aboutMe/views.py')
    urls_path = os.path.join(settings.BASE_DIR, 'aboutMe/urls.py')
    sty_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/css/style.css')
    scr_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/js/script.js')
    ba_path = os.path.join(settings.BASE_DIR, 'aboutMe/templates/base.html')
    thisWeb_path = os.path.join(settings.BASE_DIR, 'aboutMe/templates/thisWebPage.html')
    try:
        with open(this_web_en_file_path, 'r', encoding='utf-8') as this_web_file:
            this_web_en = this_web_file.read()
            this_web_en = this_web_en.replace('\n', '<br>')
            this_web_en = this_web_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        this_web_en = "error: file not found"
    try:
        with open(views_path, 'r', encoding='utf-8') as views_file:
            views = views_file.read()
    except FileNotFoundError:
        views = "error: file not found"
    try:
        with open(urls_path, 'r', encoding='utf-8') as urls_file:
            urls = urls_file.read()
    except FileNotFoundError:
        urls = "error: file not found"
    try:
        with open(sty_path, 'r', encoding='utf-8') as sty_file:
            sty = sty_file.read()
    except FileNotFoundError:
        sty = "error: file not found"
    try:
        with open(scr_path, 'r', encoding='utf-8') as scr_file:
            scr = scr_file.read()
    except FileNotFoundError:
        scr = "error: file not found"
    try:
        with open(ba_path, 'r', encoding='utf-8') as ba_file:
            ba = ba_file.read()
    except FileNotFoundError:
        ba = "error: file not found"
    try:
        with open(thisWeb_path, 'r', encoding='utf-8') as thisWeb_file:
            thisWeb = thisWeb_file.read()
    except FileNotFoundError:
        thisWeb = "error: file not found"
    return render(request, 'thisWebPage.html', {'this_web_en': this_web_en, 
                                                'views': views, 
                                                'urls': urls,
                                                'sty': sty, 
                                                'scr': scr, 
                                                'ba': ba, 
                                                'thisWeb': thisWeb, })

def unowar(request):
    UnoWarIntro_en_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/UnoWarIntro_en.txt')
    UWAITest_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/AITest.java')
    UWCardPileTest_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/CardPileTest.java')
    UWCardTest_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/CardTest.java')
    UWDeckTest_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/DeckTest.java')
    UWHandTest_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/HandTest.java')
    UWAI_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/AI.java')
    UWBiggestCardAI_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/BiggestCardAI.java')
    UWCard_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/Card.java')
    UWCardPile_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/CardPile.java')
    UWDeck_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/Deck.java')
    UWHand_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/Hand.java')
    UWSmallestCardAI_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/SmallestCardAI.java')
    UWTournament_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/Tournament.java')
    UWUnoWarMatch_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/UnoWarMatch.java')
    try:
        with open(UnoWarIntro_en_path, 'r', encoding='utf-8') as UnoWarIntro_en_file:
            UnoWarIntro_en = UnoWarIntro_en_file.read()
            UnoWarIntro_en = UnoWarIntro_en.replace('\n', '<br>')
            UnoWarIntro_en = UnoWarIntro_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        UnoWarIntro_en = "error: file not found"
    try:
        with open(UWAITest_path, 'r', encoding='utf-8') as UWAITest_file:
            UWAITest = UWAITest_file.read()
    except FileNotFoundError:
        UWAITest = "error: file not found"
    try:
        with open(UWCardPileTest_path, 'r', encoding='utf-8') as UWCardPileTest_file:
            UWCardPileTest = UWCardPileTest_file.read()
    except FileNotFoundError:
        UWCardPileTest = "error: file not found"
    try:
        with open(UWCardTest_path, 'r', encoding='utf-8') as UWCardTest_file:
            UWCardTest = UWCardTest_file.read()
    except FileNotFoundError:
        UWCardTest = "error: file not found"
    try:
        with open(UWDeckTest_path, 'r', encoding='utf-8') as UWDeckTest_file:
            UWDeckTest = UWDeckTest_file.read()
    except FileNotFoundError:
        UWDeckTest = "error: file not found"
    try:
        with open(UWHandTest_path, 'r', encoding='utf-8') as UWHandTest_file:
            UWHandTest = UWHandTest_file.read()
    except FileNotFoundError:
        UWHandTest = "error: file not found"
    try:
        with open(UWAI_path, 'r', encoding='utf-8') as UWAI_file:
            UWAI = UWAI_file.read()
    except FileNotFoundError:
        UWAI = "error: file not found"
    try:
        with open(UWBiggestCardAI_path, 'r', encoding='utf-8') as UWBiggestCardAI_file:
            UWBiggestCardAI = UWBiggestCardAI_file.read()
    except FileNotFoundError:
        UWBiggestCardAI = "error: file not found"
    try:
        with open(UWCard_path, 'r', encoding='utf-8') as UWCard_file:
            UWCard = UWCard_file.read()
    except FileNotFoundError:
        UWCard = "error: file not found"
    try:
        with open(UWCardPile_path, 'r', encoding='utf-8') as UWCardPile_file:
            UWCardPile = UWCardPile_file.read()
    except FileNotFoundError:
        UWCardPile = "error: file not found"
    try:
        with open(UWDeck_path, 'r', encoding='utf-8') as UWDeck_file:
            UWDeck = UWDeck_file.read()
    except FileNotFoundError:
        UWDeck = "error: file not found"
    try:
        with open(UWHand_path, 'r', encoding='utf-8') as UWHand_file:
            UWHand = UWHand_file.read()
    except FileNotFoundError:
        UWHand = "error: file not found"
    try:
        with open(UWSmallestCardAI_path, 'r', encoding='utf-8') as UWSmallestCardAI_file:
            UWSmallestCardAI = UWSmallestCardAI_file.read()
    except FileNotFoundError:
        UWSmallestCardAI = "error: file not found"
    try:
        with open(UWTournament_path, 'r', encoding='utf-8') as UWTournament_file:
            UWTournament = UWTournament_file.read()
    except FileNotFoundError:
        UWTournament = "error: file not found"
    try:
        with open(UWUnoWarMatch_path, 'r', encoding='utf-8') as UWUnoWarMatch_file:
            UWUnoWarMatch= UWUnoWarMatch_file.read()
    except FileNotFoundError:
        UWUnoWarMatch= "error: file not found"
    return render(request, 'unowar.html', {'UWAITest': UWAITest, 
                                            'UWCardPileTest': UWCardPileTest,
                                            'UWCardTest': UWCardTest,
                                            'UWDeckTest': UWDeckTest,
                                            'UWHandTest': UWHandTest,
                                            'UWAI': UWAI,
                                            'UWBiggestCardAI': UWBiggestCardAI,
                                            'UWCard': UWCard,
                                            'UWCardPile': UWCardPile,
                                            'UWDeck': UWDeck,
                                            'UWHand': UWHand,
                                            'UWSmallestCardAI': UWSmallestCardAI,
                                            'UWTournament': UWTournament,
                                            'UWUnoWarMatch': UWUnoWarMatch,
                                            'UnoWarIntro_en': UnoWarIntro_en,})

def vectorCalculator(request):
    vectorIntro_en_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/vectorIntro_en.txt')
    vector3_path = os.path.join(settings.BASE_DIR, 'aboutMe/static/text/vector3.py')
    try:
        with open(vectorIntro_en_path, 'r', encoding='utf-8') as vectorIntro_en_file:
            vectorIntro_en = vectorIntro_en_file.read()
            vectorIntro_en = vectorIntro_en.replace('\n', '<br>')
            vectorIntro_en = vectorIntro_en.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    except FileNotFoundError:
        vectorIntro_en = "error: file not found"
    try:
        with open(vector3_path, 'r', encoding='utf-8') as vector3_file:
            vector3 = vector3_file.read()
    except FileNotFoundError:
        vector3 = "error: file not found"
    return render(request, 'vectorCalculator.html', {'vector3': vector3, 'vectorIntro_en': vectorIntro_en,})
