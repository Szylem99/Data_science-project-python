#!/usr/bin/env python
# coding: utf-8

# # <center> <span style='color:purple' > Projekt badawczy </center> </span>
# ## <center> *Zastosowania R i Pythona w Data Science* </center>
# ### <center> Anna, Szymon </center>

# #### Importy

# In[33]:


import pandas as pd 
from matplotlib import pyplot as plt
#!pip install plotly==5.11.0 --user
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier


# <font color='blue'> Pobranie danych, usunięcie wybranych obserwacji i weryfikacja braków </font> 

# In[17]:


sklepy = pd.read_excel('dane_projekt2022.xlsx')


# In[3]:


sklepy.head()


# In[18]:


sklepy2 = sklepy.drop(range(1300,1599))


# In[19]:


sklepy2.head()


# In[20]:


sklepy2.isnull().sum()


# Nie ma braków danych.

# #### Statystyki opisowe i wizualizacje

# In[21]:


sklepy2.describe()


# Średnie zadowolenie z asortymentu wynosi 41,7 punktów na 100. Jest to dość niski wynik, większość klientów nie jest zadowolona z asortymentu. Mediana wynosi 40. 75% klientów ocenia asortyment na co najwyżej 59 punktów. <br><br> Średnie zadowolenie z obsługi wynosi 53,1 punktów na 100. Jest to wynik wyższy, niż w przypadku asortymentu, ale nadal zadowolenie z obsługi jest dość niskie. Mediana wynosi 56. 75% klientów ocenia asortyment na co najwyżej 81 punktów. <br><br> Średnio klienci robią zakupy 4 dni w tygodniu, także w 3 trzech innych marketach poza marketem X. <br><br> Średnia odległość klientów marketu X od ich miejsca zamieszkania wynosi prawie 12 kilometrów. Taką trasę może być trudno pokonać pieszo, dlatego większość klientów zapewne porusza się samochodem, transportem publicznym itd. 

# Wykres słupkowy dla zmiennej **internet**

# In[9]:


sklepy2.internet.value_counts().plot.bar(color='lightgreen',
                                              alpha=0.8,
                                                title='Czy robi zakupy przez Internet?',
                                                xlabel='0 - nie, 1 - tak',
                                                edgecolor='navy',
                                                linewidth=3,
                                                ylabel='Liczba klientów')


# Większość respondentów nie robi zakupów przez internet.

# In[4]:


fig = px.scatter_matrix(sklepy2,
                        dimensions=sklepy2.columns[[0,1,7,10]],
                        color='plec',
                        opacity=1)

fig.show()


# Na wizualizacji widać, że obserwacje układają się w specyficzne prostokąty. <br> Bardzo wielu mężczyzn tak samo oceniło asortyment i obsługę. <br> Klienci, którzy mieszkają ok. 50-100 kilometrów od marketu stanowią dwie obserwacje odstające. Ich zadowolenie z asortymentu i obsługi jest niskie. <br> Maksymalna ocena obsługi oraz asortymentu jest wyższa dla osób od 60 roku życia w porównaniu do młodszych respondentów.

# In[5]:


fig = px.density_heatmap(sklepy, 
                         x='dochod',
                         y='pozyczka',
                         color_continuous_scale=px.colors.sequential.Sunsetdark,
                         marginal_x='histogram', 
                         text_auto=True)
                        

fig.update_layout(width=700, 
                  height=500)

fig.show()


# Najwięcej respondentów ma dochód w przedziale 2-2.1 oraz 2.2-2.3. W każdej z tych grup występuje ponad 1000 respondentów, którzy mają pożyczki. Zdecydowana większość respondentów posiada zobowiązanie w tej postaci.

# In[6]:


sklepy2.kredytowa.value_counts().plot.pie(colors=['deeppink', 'steelblue'],
                                                ylabel='',
                                     Najwięcej respondentów ma dochód w przedziale 2-2.1 oraz 2.2-2.3. W każdej z tych grup występuje ponad 1000 respondentów, którzy mają pożyczki. Zdecydowana większość respondentów posiada zobowiązanie w tej postaci.           labels=('nie', 'tak'),
                                                legend=True,
                                                labeldistance=None,
                                                autopct='%.2f%%',
                                                title='Czy posiada kartę kredytową?')


# Większość respondentów nie posiada karty kredytowej, co wraz z kształtowaniem się zmiennej "internet" trudno odnieść do obecnych czasów, kiedy płatności kartą i zakupy internetowe są bardzo popularne.

# ## Klasyfikacja 
# 3 metody, zmienna zależna: "zakupy"

# #### Klasyfikator kNN

# Wyznaczenie zmiennych niezależnych i zmiennej zależnej

# In[22]:


X = sklepy2[['asortyment','obsluga', 'dochod','wiek', 'odleglosc']]


# In[23]:


y = sklepy2['zakupy'].astype(int)
print(type(y))
print(y)


# **Algorytm klasyfikacji**

# In[24]:


# podział na zbiór uczący i testowy

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.2)


# In[25]:


# standaryzacja

scaler = StandardScaler()
scaler.fit(X_train)


# In[26]:


X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# In[27]:


# utworzenie i wytrenowanie klasyfikatora

kNN_clf = KNeighborsClassifier(n_neighbors = 3) 
kNN_clf.fit(X_train_std, y_train) 


# In[28]:


# predykcja dla zbioru testowego

y_pred = kNN_clf.predict(X_test_std)
y_pred


# **Ocena klasyfikatora**

# In[29]:


# accuracy

kNN_clf.score(X_test_std, y_test)


# Niecałe 80% obserwacji zostało sklasyfikowane poprawnie. <br> Accuracy (dokładność) nie jest wystarczającą miarą, dlatego warto zwrócić uwagę na pozostałe wskaźniki.

# In[30]:


# macierz pomyłek

print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(kNN_clf, X_test_std, y_test)


# Zdecydowana większość obserwacji została rozpoznana prawidłowo. 

# In[31]:


# precyzja, czułość i f1

print(classification_report(y_test, y_pred))


# **0 - nie zrobi w przyszłości zakupów w markecie X** <br>
# Wśród klientów sklasyfikowanych do grupy 0, 80% to osoby, które w rzeczywistości nie zrobią zakupów w tym markecie. <br> 82% klientów, którzy zadeklarowali, że nie zrobią zakupów w tym sklepie, zostało słusznie sklasyfikowanych do grupy 0. <br> Średnia harmoniczna z dwóch miar wynosi 0.81. <br> Dla grupy 0 mierniki osiągają wysokie wartości. Predykcja w większości przebiegła prawidłowo.
# 
# **1 - zrobi w przyszłości zakupy w markecie X** <br>
# Klienci, którzy zostali sklasyfikowani do grupy 1, w 80% w rzeczywistości zadeklarowali, że zrobią jeszcze zakupy w markecie X. <br> 78% klientów, którzy zadeklarowali, że zrobią zakupy w tym sklepie, zostało słusznie sklasyfikowanych do grupy 1. <br> Średnia harmoniczna, czyli f1, wynosi 79%. <br> Dla grupy 1 jakość predykcji również można uznać za wysoką.

# #### Drzewo klasyfikacyjne ze strojeniem maksymalnej głębokości

# In[34]:


dt_clf = DecisionTreeClassifier(criterion='gini', 
                                max_depth=2)
dt_clf.fit(X_train_std, y_train)


# In[35]:


my_param = GridSearchCV(dt_clf, 
                         param_grid={'max_depth':range(1,20)},
                         cv=5, 
                         scoring='accuracy')


# In[36]:


my_param.fit(X_train_std, y_train)


# In[37]:


print(my_param.best_params_)
print(my_param.best_score_)


# Przeszukanie siatki wskazało, że najlepsza **maksymalna głębokość** dla drzewa to **6**. W takiej sytuacji wybrane kryterium, czyli miara 'accuracy', wynosi 0,83. <br><br> Powtórzenie algorytmu z zastosowaniem wybranego hiperparametru.

# In[38]:


# utworzenie i wytrenowanie klasyfikatora

dt_clf = DecisionTreeClassifier(criterion='gini', 
                                max_depth=6)
dt_clf.fit(X_train_std, y_train)


# In[39]:


# predykcja dla zbioru testowego

y_pred2 = dt_clf.predict(X_test_std)


# In[40]:


# macierz pomyłek

print(confusion_matrix(y_test, y_pred2))
plot_confusion_matrix(dt_clf, X_test_std, y_test)


# Na podstawie macierzy pomyłek można zauważyć, że większość obserwacji została rozpoznana poprawnie.

# In[41]:


# ocena klasyfikatora

print(classification_report(y_test, y_pred2))


# **0 - nie zrobi w przyszłości zakupów w markecie X** <br>
# Wśród klientów sklasyfikowanych do grupy 0, 80% to osoby, które w rzeczywistości nie zrobią zakupów w tym markecie. <br> 88% klientów, którzy zadeklarowali, że nie zrobią zakupów w tym sklepie, zostało słusznie sklasyfikowanych do grupy 0. <br> Średnia harmoniczna z dwóch miar wynosi 0.84. <br> Dla grupy 0 mierniki osiągają wysokie wartości. Predykcja w większości przebiegła prawidłowo.
# 
# **1 - zrobi w przyszłości zakupy w markecie X** <br>
# Klienci, którzy zostali sklasyfikowani do grupy 1, w 86% w rzeczywistości zadeklarowali, że zrobią jeszcze zakupy w markecie X. <br> 77% klientów, którzy zadeklarowali, że zrobią zakupy w tym sklepie, zostało słusznie sklasyfikowanych do grupy 1. <br> Średnia harmoniczna, czyli f1, wynosi 81%. <br> Dla grupy 1 jakość predykcji również można uznać za wysoką.
# 
# Jakość predykcji jest bardzo zbliżona do tej uzyskanej za pomocą metody najbliższego sąsiada.

# In[42]:


ROC = plot_roc_curve(dt_clf, X_test_std, y_test)


# Pole pod krzywą ROC wynosi 0,85 i oznacza dobre dopasowanie modelu do danych rzeczywistych. 

# #### Las losowy

# In[43]:


# stworzenie i wytrenowanie klasyfikatora, 220 drzew, obliczenia wykonywane równolegle

rf_clf = RandomForestClassifier(n_estimators=220, n_jobs=-1)
rf_clf.fit(X_train_std, y_train)


# In[44]:


# predykcja dla zbioru testowego

y_pred3 = rf_clf.predict(X_test_std)


# In[45]:


# macierz pomyłek

plt.figure(figsize=(5,5))
plot_confusion_matrix(rf_clf, X_test_std, y_test)


# Macierz pomyłek przedstawia większość prawidłowo sklasyfikowanych obserwacji.

# In[46]:


# ocena klasyfikatora

print(classification_report(y_test, y_pred3))


# **0 - nie zrobi w przyszłości zakupów w markecie X** <br>
# Wśród klientów sklasyfikowanych do grupy 0, 81% to osoby, które w rzeczywistości nie zrobią zakupów w tym markecie. <br> 87% klientów, którzy zadeklarowali, że nie zrobią zakupów w tym sklepie, zostało słusznie sklasyfikowanych do grupy 0. <br> Średnia harmoniczna z dwóch miar wynosi 84%. <br> Dla grupy 0 mierniki osiągają bardzo wysokie wartości.
# 
# **1 - zrobi w przyszłości zakupy w markecie X** <br>
# Klienci, którzy zostali sklasyfikowani do grupy 1, w 85% w rzeczywistości zadeklarowali, że zrobią jeszcze zakupy w markecie X. <br> 79% klientów, którzy zadeklarowali, że zrobią zakupy w tym sklepie, zostało słusznie sklasyfikowanych do grupy 1. <br> Średnia harmoniczna, czyli f1, wynosi 82%. <br> Dla grupy 1 jakość predykcji również można uznać za wysoką.

# <font color='#eb3663'> Podsumowanie dla klasyfikacji </font>
# 
# Dla każdej z metod jakość predykcji była bardzo wysoka. Najniższa wartość miernika, jaka pojawiła się w raporcie, to 77% i dotyczy czułości dla grupy 1. <br> 
# Najniższe wyniki otrzymano dla pierwszej metody (najbliższego sąsiada), choć różnice w jakości zastosowanych metod były znikome. Bardzo podobnie kształtuje się jakość predykcji dla drzewa klasyfikacyjnego oraz lasu losowego. Dla dwóch metod wskaźnik f1 (harmoniczna średnia precyzji i czułości) dla grupy 1 wynosi 84%. Zaleca się zastosowanie drzewa klasyfikacyjnego lub lasu losowego, choć należy podkreślić, że metoda najbliższego sąsiada także daje bardzo zadowalające wyniki.

# ### Regresja

# #### Regresja klasyczna

# W naszych modelach regresyjnych wzięliśmy pod uwagę 4 zmienne

# In[41]:


X = sklepy2[['asortyment', 'dochod','wiek', 'odleglosc']]


# In[42]:


y = sklepy2['obsluga'].astype(int)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)


# In[44]:


lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)


# In[45]:


print('Współczynniki przy zmiennych:',lm.coef_)
print('Wyraz wolny:',lm.intercept_)


# In[46]:


mse = mean_squared_error(y_test,y_pred)
rmse = mse ** (1 / 2)
print('MSE:', mse)
print('RMSE:', rmse)


# Wysoka wartość RMSE wskazuje, że w trakcie przewidywania zadowolenia z obsługi, pomiary mylą się o przeciętnie o 19,6 punktu. Jest to wynik niezadowalający, który wskazuje, że, by stworzyć model, który będzie poprawnie przewidywał poziom zadowolenia, należy wykorzystać inne metody.

# In[47]:


y_mean = y_test.mean()
print(rmse/y_mean * 100)


# In[48]:


r2 = r2_score(y_test, y_pred)
print(r2)


# Współczynik determinacji równy 0.61 również wskazuje na to, że dopasowanie modelu do danych rzeczywistych jest średnie.

# In[50]:


plt.figure(figsize=[6,6])
plt.scatter(y_test,y_pred,
           color='green')
plt.plot(y_test, y_test, 
         color='orange')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()


# In[51]:


permutated_importances = permutation_importance(lm, X_test, y_test, random_state=23, 
                                                scoring='neg_mean_squared_error')


# In[53]:


features_perm = pd.DataFrame(permutated_importances.importances_mean, index=X.columns, columns=['Importances'])
features_perm.sort_values(by=['Importances'], inplace=True)
plt.barh(features_perm.index, features_perm['Importances'], color='hotpink')
plt.title('Permutation based feature importances')
plt.show()


# Najwększy wkład w model mają zmienne dochód i asortyment. Wpływ wieku jest minimalny, a samej odległości praktycznie zerowy. 
# Zgadza się to z naszymi wstępnymi intuicjami, według których największych wpływ na zadowolenie miała mieć możliwość wyboru 
# oraz własna sytuacja ekonomiczna. Wiek i odległość nie mają żadnego interpretowalnego przełożenia na ocenę obsługi co znalazło odzwierciedlenie w samym modelu.
# 

# #### Drzewa regresyjne

# In[54]:


X = sklepy2[['asortyment', 'dochod','wiek', 'odleglosc']]


# In[55]:


y = sklepy2['obsluga'].astype(int)


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)


# In[57]:


dt_reg = DecisionTreeRegressor(max_depth=3)
dt_reg.fit(X_train,y_train)
y_pred = dt_reg.predict(X_test)


# In[58]:


plt.figure(figsize=(15,12)) # ustawienie rozmiaru rysunku
figure = plot_tree(dt_reg, 
                   feature_names=X.columns, 
                   filled=True, 
                   rounded=True) 


# W przypadku drzewa regresyjnego o głębokości 3, zmiennymi, które decydowały o podziale obserwacji na grupy, były wyłącznie zmienne asortyment oraz dochód. Najliczniejsza grupa cechuje się wskaźnikiem dochodu niższym niż 2,85 i zadowoleniem z asortymentu wyższym niż 51,5. Najmniej liczną grupę stanowią klienci, mający dochód między 2,85 i 2,95, których poziom zadowolenia z asortymentu był niższy niż 20.5. W przypadku tej grupy squared_error na poziomie 583, wskazuje na to, że regresja obarczona jest ogromnym błędem.

# In[22]:


mse =mean_squared_error(y_test,y_pred)
rmse = mse ** (1 / 2)
print('MSE:', mse)
print('RMSE:', rmse)


# RMSE równe 10.4 wskazuje, że wartości przewidywane różnią się przeciętnie o 10.4 względem wartości rzeczywistych jest to o wiele 

# In[23]:


features = pd.DataFrame(dt_reg.feature_importances_, index=X.columns, columns=['Importances'])
features.sort_values(by=['Importances'], inplace=True)
plt.barh(features.index, features['Importances'], color='blue')
plt.title('Feature importances')
plt.show()


# W przypadku drzewa regresyjnego wpływ na zmienną mają wyłącznie zmienne dochód i asortyment. Marginalny, w przypadku regresji klasycznej, wpływ odległości i wieku tym razem nie jest odczuwalny.

# #### Las losowy

# W przypadku lasu losowego zastosowaliśmy metodę strojenia parametrów

# In[59]:


X = sklepy2[['asortyment', 'dochod','wiek', 'odleglosc']]


# In[25]:


y = sklepy2['obsluga'].astype(int)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[28]:


rf_reg = RandomForestRegressor()


# In[29]:


nastrojony = GridSearchCV(rf_reg, 
                         param_grid={'max_depth':range(1,20),
                                    'min_samples_leaf':[2,4,6],
                                    'min_samples_split':[2,4,6],
                                    'n_estimators':[10,20],
                                    "bootstrap": [True, False]},
                          scoring='neg_mean_squared_error',
                         cv=5 
                         )
nastrojony.fit(X_train, y_train)


# In[30]:


print(nastrojony.best_params_)


# Okazało sie, że najlepiej dopasowany model tworzony jest dla lasu losowego o glębokości 7, włączonym bootstrapie, minimalnej liczbie obserwacji w liściu równej 2, minimalnej liczbie obserwacji w węźle równej 4 i liczbą drzew równą 10.

# In[32]:


rf_reg = RandomForestRegressor(max_depth=7, 
                                min_samples_leaf=2,
                               min_samples_split=4,
                               n_estimators=10,
                               bootstrap=True)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)                              


# In[33]:


mse = mean_squared_error(y_test,y_pred)
rmse = mse ** (1 / 2)
print('MSE:', mse)
print('RMSE:', rmse)


# RMSE 5,8 oznacza, że wartości przewidywane mylą się przeciętnie o 5,8 od wartości rzeczywistych. Jest to zdecydowanie najlepszy wynik wśród wszystkich 3 metod regresyjnych, które zastosowaliśmy

# In[60]:


y_mean = y_test.mean()
print(rmse/y_mean * 100)


# In[35]:


rf_reg.score(X_test, y_test)


# Współczynnik determinacji równy 0,97 wskazuje na doskonałe dopasowanie modelu do danych rzeczywistych

# In[61]:


features = pd.DataFrame(rf_reg.feature_importances_, 
                        index=X.columns, columns=['Importances'])
features.sort_values(by=['Importances'], inplace=True)
plt.barh(features.index, features['Importances'], color='blue')
plt.title('Feature importances')
plt.show()


# W przypadku ważności wybranych zmiennych, las losowy nie różni się niczym od poprzednich metod - dochód i asortyment mają największe znaczenie w przypadku przewidywania zadowolenia z obsługi. 

# Podsumowanie

# W przypadku metod regresyjnych niewątpliwie najlepszą metodą, pozwalającą przewidywać zadowolenie z obsługi jest las losowy, który dzięki doskonałemu R^2 i niskiej wartości RMSE cechuje się najlepszą jakością spośród wszystkich badanych modeli regresji.

# In[ ]:




