# Generated by Django 4.1.7 on 2023-04-26 17:08

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('skin', '0012_register_form'),
    ]

    operations = [
        migrations.RenameField(
            model_name='register_form',
            old_name='password',
            new_name='password1',
        ),
        migrations.RenameField(
            model_name='register_form',
            old_name='password_confirmation',
            new_name='password2',
        ),
    ]
