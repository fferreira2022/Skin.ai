# Generated by Django 4.1.7 on 2023-04-01 09:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('skin', '0002_topic_room_host_message_room_topic'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='room',
            options={'ordering': ['-updated', '-created']},
        ),
        migrations.RenameField(
            model_name='message',
            old_name='message_content',
            new_name='body',
        ),
    ]
