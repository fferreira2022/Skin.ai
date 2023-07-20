from django.contrib import admin

# Register your models here.

from .models import Room, Topic, Message, Skin_lesion_analysis

admin.site.register(Room)
admin.site.register(Topic)
admin.site.register(Message)
admin.site.register(Skin_lesion_analysis)


