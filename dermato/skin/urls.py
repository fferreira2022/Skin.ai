from django.urls import path
from . import views
from django.contrib.auth import views as auth_views # <= à importer pour processus de réinitialisation de mots de passe

urlpatterns = [
    path('login/', views.loginPage, name='login'),
    path('logout/', views.logoutUser, name='logout'),
    path('register/', views.signup, name='register'),
    
    path('terms_acceptance', views.terms_acceptance, name='terms_acceptance'),
    
    path('', views.home, name='home'),
    path('profile/<str:pk>/', views.userProfile, name='user-profile'),
    path('update_profile/', views.updateProfile, name='update-profile'),
    path('room/<str:pk>/', views.room, name='room'),
    path('create-room/', views.createRoom, name='create-room'),
    path('update-room<str:pk>/', views.updateRoom, name='update-room'),
    path('delete-room<str:pk>/', views.deleteRoom, name='delete-room'),
    path('delete-message<str:pk>/', views.deleteMessage, name='delete-message'),
    path('upload_image/', views.uploadImage, name='upload-image'),
    path('about/', views.about, name='about'),
    path('disclaimer/', views.disclaimer, name='disclaimer'),
    path('contact/', views.contact, name='contact'),
    path('faq/', views.faq, name='faq'),
    path('forum/', views.forum, name='forum'),
    
    path('activate/<uidb64>/<token>/', views.activate, name='activate'),
    
    # urls pour la réinitialisation du mot de passe
    path('reset_password/', auth_views.PasswordResetView.as_view(template_name="skin/password_reset.html"), name='reset_password'),
    path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name="skin/password_reset_sent.html"), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="skin/password_reset_form.html"), name='password_reset_confirm'),
    path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view(template_name="skin/password_reset_done.html"), name='password_reset_complete'),
    
    # urls pour l'implémentation de stripe
    path('checkout/', views.CreateCheckoutSession, name='create-checkout-session'),
    path('checkout_success/', views.success, name='checkout_success'),
    path('checkout_cancelation/', views.cancelation, name='checkout_cancelation'),
    
    # url pour le chatbot
    path('chatbot/', views.chatbot, name='chatbot'),
]

