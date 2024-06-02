from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='home'),
    path('proceed',views.proceed,name='proceed'),
    path('modelselection',views.modelselection,name='modelselection'),
    path('predict',views.predict,name='predict'),
    path('graph',views.graph,name='graph')



]