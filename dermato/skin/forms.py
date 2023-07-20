from django.forms import ModelForm
from .models import Room
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User
from django import forms

from django.core.mail import send_mail
from django.conf import settings

class RoomForm(ModelForm):
    class Meta:
        model = Room
        fields = ['topic','name', 'description']

    def __init__(self, *args, **kwargs):
        super(RoomForm, self).__init__(*args, **kwargs)
        self.fields['topic'].widget.attrs['class'] = 'form-control'
        self.fields['name'].widget.attrs['class'] = 'form-control'
        self.fields['name'].widget.attrs['placeholder'] = 'Titre'
        self.fields['description'].widget.attrs['class'] = 'form-control'
        self.fields['description'].widget.attrs['placeholder'] = 'Message'
            
class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super(SignUpForm, self).__init__(*args, **kwargs)
        self.fields['username'].widget.attrs['class'] = 'form-control'
        self.fields['username'].widget.attrs['placeholder'] = 'Nom d\'utilisateur'
        self.fields['email'].widget.attrs['class'] = 'form-control'
        self.fields['email'].widget.attrs['placeholder'] = 'Adresse email'
        self.fields['password1'].widget.attrs['class'] = 'form-control'
        self.fields['password1'].widget.attrs['placeholder'] = 'Mot de passe'
        self.fields['password2'].widget.attrs['class'] = 'form-control'
        self.fields['password2'].widget.attrs['placeholder'] = 'Confirmer mot de passe'
        
        
class UpdateProfileForm(UserChangeForm):
    username = forms.CharField(max_length=200, widget= forms.TextInput(attrs={'class': 'form-control'}), required=False)
    email = forms.EmailField(max_length=200, widget= forms.TextInput(attrs={'class': 'form-control'}), required=False)
    class Meta:
        model = User
        fields = ['username', 'email']
    


class ContactForm(forms.Form):
    subject = forms.CharField(max_length=200, widget= forms.TextInput(attrs={'class': 'form-control'}), required=True)
    message = forms.CharField(widget= forms.Textarea(attrs={'class': 'form-control'}), required= True)
    from_email = forms.EmailField(widget= forms.EmailInput(attrs={'class': 'form-control'}), required=True)



    

    
        
      
    
        