# Generated by Django 4.1.7 on 2023-04-26 15:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('skin', '0009_delete_register_form'),
    ]

    operations = [
        migrations.CreateModel(
            name='Register_form',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=200)),
                ('email', models.EmailField(max_length=200)),
                ('password', models.CharField(max_length=200)),
                ('password_confirmation', models.CharField(max_length=200)),
            ],
        ),
    ]
