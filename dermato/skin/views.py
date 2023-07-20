import os
from typing import Any, Dict
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import Room, Topic, Message, Skin_lesion_analysis
from .forms import RoomForm, SignUpForm, ContactForm, UpdateProfileForm
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.hashers import check_password
from django import forms
from django.core.mail import send_mail, BadHeaderError
from django.conf import settings

from django.template.loader import render_to_string
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.core.mail import EmailMessage

from .tokens import account_activation_token
import uuid 

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from .process_image import wisdom_of_the_crowd
from .svm_classifier import predict_image_category


from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.shortcuts import render, redirect
from django.utils import timezone
import imghdr

from django.core.paginator import Paginator

import stripe
from django.views import View
from django.views.generic import TemplateView
from django.urls import reverse

import openai
openai.api_key = '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'

import cv2
import numpy as np

# implémentation du système de paiement Stripe
@login_required(login_url='login') 
def CreateCheckoutSession(request):
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    checkout_session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[
            {
                'price': 'price_1NHSI0KFSQgUiiU0gVXXFbOy',
                'quantity': 1,
            },
        ],
        mode='payment',
        success_url = request.build_absolute_uri(reverse('checkout_success')) + '?session_id={CHECKOUT_SESSION_ID}',
        cancel_url = request.build_absolute_uri(reverse('checkout_cancelation'))
        )
    
    context = {
        'session_id': checkout_session.id,
        'STRIPE_PUBLIC_KEY': settings.STRIPE_PUBLIC_KEY
    }
    print(settings.STRIPE_PUBLIC_KEY)
    return render(request, 'skin/checkout.html', context)

        
def success(request):
    return render(request, 'skin/checkout_success.html')

def cancelation(request):
    return render(request, 'skin/checkout_cancelation.html')



# activation du compte après inscription, fait appel au fichier tokens.py
def activate(request, uidb64, token):
    user = request.user
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except:
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        
        messages.success(request, f'Votre compte a bien été activé')
        return redirect('login')
    else:
        messages.error(request, "Le lien d'activation n'est pas valide")
    return redirect('home')

# confirmation de l'adresse email
def confirm_Email(request, user, to_email):
    mail_subject = 'Confirmer votre adresse email'
    message = render_to_string('skin/confirm_email.html', {
        'user': user,
        'domain': get_current_site(request).domain,
        'uid': urlsafe_base64_encode(force_bytes(user.pk)),
        'token': account_activation_token.make_token(user),
        'protocol': 'https' if request.is_secure() else 'http',
    })
    email = EmailMessage(mail_subject, message, to=[to_email])
    if email.send():
        messages.success(request, f'Cher {user.username}, veuillez confirmer votre adresse email en cliquant sur le lien activation.')
    else:
        messages.error(request, f"Une erreur est survenue lors de l'envoi du lien de confirmation à {to_email} ")
    # return render(request,'skin/confirm_email.html', {'email': email})

# formulaire d'inscription
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.is_active = False
            user.save()
            confirm_Email(request, user, form.cleaned_data.get('email'))
            return redirect('home')
        else:
            messages.error(request, "Une erreur est survenue lors de l'inscription")
    else:
        form = SignUpForm()
        
    return render(request = request, template_name='skin/register.html', context={'form': form})


# formulaire de connexion
def loginPage(request):
    page = 'login'
    # si l'utilisateur est déjà connecté on le redirige vers la page d'accueil
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        email = request.POST.get('email').lower()
        password = request.POST.get('password')
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            messages.error(request, 'Adresse électronique invalide')
            return redirect('login')
        if user.check_password(password):
            # Si le mot de passe est correct, on connecte l'utilisateur
            login(request, user)
            return redirect('home')
        else:
            # si le mot de passe est incorrect, on affiche un message d'erreur
            messages.error(request, 'Nom d\'utilisateur ou mot de passe invalide')
            return redirect('login')
    context ={'page': page}
    return render(request, 'skin/login_register.html', context)


def logoutUser(request):
    logout(request)
    return redirect('home')


def home(request):
    return render(request, 'skin/home.html')


def room(request, pk):
    room = Room.objects.get(id=pk)
    
    paginator = Paginator(room.message_set.all(), 5)
    page = request.GET.get('page')
    room_messages = paginator.get_page(page)

    #room_messages = room.message_set.all()#.order_by('-created') # ici order_by permet d'ordonner les messages en fonction de leur date de création
    participants = room.participants.all()
    if request.method == 'POST':
        message = Message.objects.create(
            user = request.user,
            room = room,
            body = request.POST.get('body')
        )
        room.participants.add(request.user)
        return redirect('room', pk=room.id)
    context = {'room': room, 'room_messages': room_messages, 'participants': participants}
    return render(request, 'skin/room.html', context)


# profil utilisateur
def userProfile(request, pk):
    user = User.objects.get(id=pk)
    email = user.email
    password = user.password
    rooms = user.room_set.all()
    room_messages = user.message_set.all()
    topics = Topic.objects.all()
    
    result_history = Skin_lesion_analysis.objects.filter(user=request.user)
    paginator = Paginator(result_history, 3)  # on affiche 3 résultats par page
    page_number = request.GET.get('page')
    result_page = paginator.get_page(page_number)
    
    context = {'user': user, 'email': email, 'password' : password, 'rooms': rooms, 'room_messages': room_messages, 'topics': topics, 'result_page': result_page,}
    return render(request, 'skin/profile.html', context)


# modification du profil utilisateur
@login_required(login_url='login') 
def updateProfile(request):
    current_user = User.objects.get(id=request.user.id)
    form = UpdateProfileForm(instance=current_user)
    if request.method == 'POST':
        form = UpdateProfileForm(request.POST, instance=current_user)
        if form.is_valid():
            form.save()
            login(request, current_user)
            messages.success(request, 'Votre profil a bien été mis à jour')
    context = {'form': form, 'user': current_user}
    return render(request, 'skin/update_profile.html', context)


# création d'un nouveau fil de discussion sur le forum
@login_required(login_url='login') 
def createRoom(request):
    form = RoomForm()
    
    if request.method == 'POST':
        form = RoomForm(request.POST)
        if form.is_valid():
            room = form.save(commit=False)
            room.host = request.user
            room.save()
            return redirect('forum')
    context = {'form': form}
    return render(request, 'skin/room_form.html', context)


# modifier un fil de discussion 
@login_required(login_url='login') 
def updateRoom(request, pk): # pk = primary key
    room = Room.objects.get(id=pk)
    form = RoomForm(instance=room)
    
    if request.user != room.host:
        return HttpResponse('You are not authorized to edit this room')
    
    if request.method == 'POST':
        form = RoomForm(request.POST, instance=room)
        if form.is_valid():
            form.save()
            return redirect('forum')
    context = {'form': form}
    return render(request, 'skin/room_form.html', context)

# supprimer un fil de discussion 
@login_required(login_url='login') 
def deleteRoom(request, pk):
    room = Room.objects.get(id=pk)
    
    if request.user != room.host:
        return HttpResponse('You are not authorized to edit this room')
    
    if request.method == 'POST':
        room.delete()
        return redirect('forum')
    return render(request, 'skin/delete.html', {'obj': room})


# supprimer un message
@login_required(login_url='login') 
def deleteMessage(request, pk):
    message = Message.objects.get(id=pk)
    
    if request.user != message.user:
        return HttpResponse('You are not authorized to delete this message')
    
    if request.method == 'POST':
        message.delete()
        return redirect('forum')
    return render(request, 'skin/delete.html', {'obj': message})


def is_valid_image_type(file):
    valid_extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.gif', '.tiff']
    _, file_extension = os.path.splitext(file.name)
    file_extension = file_extension.lower()
    if file_extension in valid_extensions:
        return True
    file_type = imghdr.what(file)
    return file_type is not None and file_type.lower() in ['jpeg', 'jpg', 'png', 'bmp', 'gif', 'tiff']

def is_skin_lesion(file):
    image_scan = predict_image_category(file)
    return True if image_scan == 'skin_lesion' else False


@login_required
def uploadImage(request):
    if request.method == 'POST':
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        user_id = request.user.id
        photo = request.FILES['image']
        if not photo:
            messages.error(request, "Veuillez sélectionner une photo.")
            return redirect('upload-image')
        if not is_valid_image_type(photo):
            messages.error(request, "Le fichier sélectionné n'est pas valide.")
            return redirect('upload-image')
        if predict_image_category(photo) == 'coco':
            messages.error(request, "La photo sélectionnée n'est pas une image de lésion cutanée. Veuillez choisir un autre fichier.")
            return redirect('upload-image')
        _, extension = os.path.splitext(photo.name)
        photo_name = f'user_id_{user_id}_{timestamp}{extension}'
        path = os.path.join('media', photo_name)
        with open(path, 'wb+') as destination:
            for chunk in photo.chunks():
                destination.write(chunk)
        prediction = wisdom_of_the_crowd(path)
        result = Skin_lesion_analysis(user=request.user, image=photo_name, body=prediction)
        result.save()
        context = {'result': result}
        return render(request, 'skin/result.html', context)
    return render(request, 'skin/upload_image.html')

def about(request):
    return render(request, 'skin/about.html')

def disclaimer(request):
    return render(request, 'skin/disclaimer.html')

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            if subject and message:
                try:
                    send_mail(subject, message, 'frederic.ferreira66@gmail.com', ["frederic.ferreira66@yahoo.com"])
                except BadHeaderError:
                    return HttpResponse("Entête invalide.")
                messages.success(request, "Votre message a été envoyé. Nous vous répondrons dès que possible.")
                return HttpResponseRedirect("/contact")
        else:
            return HttpResponse("Assurez-vous que tous les champs sont remplis.")
    else:
        form = ContactForm()
    return render(request, 'skin/contact.html', {'form': form})
    

def faq(request):
    return render(request, 'skin/faq.html')


def forum(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    
    rooms_paginator = Paginator(Room.objects.filter(Q(topic__name__icontains=q) | # le Q permet d'enchaîner les conditions, le | signifie 'ou' et peut être remplacé par '&' si l'on veut que toutes les conditions soient réunies
                                Q(name__icontains=q) | # le i de icontains permet de ne pas tenir compte de la casse
                                Q(description__icontains=q)), 8)
    page = request.GET.get('page')
    rooms = rooms_paginator.get_page(page)
    
    # rooms = Room.objects.filter(Q(topic__name__icontains=q) | # le Q permet d'enchaîner les conditions, le | signifie 'ou' et peut être remplacé par '&' si l'on veut que toutes les conditions soient réunies
    #                             Q(name__icontains=q) | # le i de icontains permet de ne pas tenir compte de la casse
    #                             Q(description__icontains=q))
    
    topics = Topic.objects.all()
    # room_count = rooms.count()
    room_messages = Message.objects.filter(Q(room__topic__name__icontains=q)) # le filtre récupère seulement l'activité concernant le room (fil de discussion) en question
    context = {'rooms': rooms, 'topics': topics, 'room_messages': room_messages}
    return render(request, 'skin/forum.html', context)


@login_required
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        message_history = []
        if 'message_history' in request.session:
            message_history = request.session['message_history'][-6:]
        message_history.append({"role": "user", "content": message})
        print(message_history)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", 
                 "content": "Tu es docteur Chad, un dermatologue. Tu travailles pour Skin.ai. Tu ne dois pas dire que tu as été créé par OpenAI.",
                
                },
                *message_history 
            ]
        )
        print(response)
        bot_reply = response.choices[0]['message']['content']
        print(bot_reply)
        message_history.append({"role": "assistant", "content": bot_reply})
        request.session['message_history'] = message_history[-7:]
    else:
        bot_reply = ""
        request.session.pop('message_history', None)
    return render(request, 'skin/chatbot.html', {'bot_reply': bot_reply})



def terms_acceptance(request):
    return render(request, 'skin/terms_acceptance.html')
        



