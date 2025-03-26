from django.urls import path
from . import views

urlpatterns = [
    path('', views.base, name='home'),  # 首页
    path('blog/', views.blog, name='blog'),  # blog
    path('contact/', views.contact, name='contact'),  # contact
    path('experience/', views.experience, name='experience'),  # experience
    path('project/', views.project, name='project'),  # project
    path('vex/', views.vex, name='vex'),  # vex
    path('discordMusicPlayerBot/', views.discordMusicPlayerBot, name='discordMusicPlayerBot'),  # discordMusicPlayerBot
    path('othello/', views.othello, name='othello'),  # othello
    path('sokoban/', views.sokoban, name='sokoban'),  # sokoban
    path('thisWebPage/', views.thisWebPage, name='thisWebPage'),  # thisWebPage
    path('unowar/', views.unowar, name='unowar'),  # unowar
    path('vectorCalculator/', views.vectorCalculator, name='vectorCalculator'),  # vectorCalculator
]
