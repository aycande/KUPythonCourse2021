#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:29:02 2021

@author: denizaycan
"""



import random

class Stock():
    def __init__(self,amount,name):
        self.name= name
        self.price= amount
        
class Bond(Stock):
    pass

    
class MutualFund():
    def __init__(self,name):
        self.name=name
        self.price=1
        

    
class Portfolio():
    def __init__(self):
        self.cash= 0
        self.stock= {}
        self.fund={}
        self.bond={}
        self.resume=[]
        
        
    def addCash(self,amount):
        self.cash += amount
        self.resume.append("$ " + str(amount)+ " added to the cash account.")
        
    def withdrawCash (self,amount):
        if self.cash <= amount:
            print("You cannot withdraw this amount. Try a lower amount.")
        else: self.cash -= amount
        self.resume.append("$ " + str(amount)+ " withdrewn from the cash account.")
    
    def buyStock(self,amount,s):
        if self.cash <= amount * s.price:
            print("You cannot buy this amount of stocks. Try a lower amount.")
        else:
            self.cash = self.cash - (amount * s.price)
            if s.name in self.stock:
                self.stock[s.name]+= amount  
            else: self.stock[s.name]= amount
            #stoklarını tuttuğun dictionaryde
            #istediğimz stockun name attribute una karşılık olarak aldığımız 
            #share sayısını yazıyoruz. arçelik(s.name)= 3(amount)
            self.resume.append("Bought "+ str(amount) + " shares of " + str(s.name)+ " by $ "+ str(amount*s.price))
    
    def sellStock(self,amount,s):
        if self.stock[s.name]<= amount:
            print("You cannot sell this amount of stocks. Try a lower amount.")
        else: 
            self.cash += (amount*s.price * random.uniform(0.5,1.5))
            self.stock[s.name] -= amount
        self.resume.append("Sold "+ str(amount) + " shares of " + str(s.name)+ " by $ "+ str(amount*s.price))
    
    def buyMutualFund(self,amount,a):
         if self.cash <= amount * a.price:
            print("You cannot buy this amount of funds. Try a lower amount.")
         else:
            self.cash = self.cash - (amount * a.price)
            if a.name in self.fund:
                self.fund[a.name]+= amount  
            else: self.fund[a.name]= amount
            self.resume.append("Bought "+ str(amount) + " shares of " + str(a.name)+ " by $ "+ str(amount*a.price))
            
    def sellMutualFund(self,amount,a):
         if self.fund[a.name] <= amount:
            print("You cannot sell this amount of funds. Try a lower amount.")
         else: 
            self.cash += (amount*a.price * random.uniform(0.9,1.2))
            self.fund[a.name] -= amount
         self.resume.append("Sold "+ str(amount) + " shares of " + str(a.name)+ " by $ "+ str(amount*a.price))
         
    def buyBond(self,amount,b):
         if self.cash <= amount * b.price:
            print("You cannot buy this amount of bonds. Try a lower amount.")
         else:
            self.cash = self.cash - (amount * b.price)
            if b.name in self.bond:
                self.bond[b.name]+= amount  
            else: self.bond[b.name]= amount
            self.resume.append("Bought "+ str(amount) + " shares of " + str(b.name)+ " by $ "+ str(amount*b.price))
    
    def sellBond(self,amount,b):
        if self.bond[b.name]<= amount:
            print("You cannot sell this amount of bonds. Try a lower amount.")
        else: 
            self.cash += (amount*b.price * random.uniform(0.5,1.5))
            self.bond[s.name] -= amount    
            self.resume.append("Sold "+ str(amount) + " shares of " + str(b.name)+ " by $ "+ str(amount*b.price))
    
    def print_portfolio(self):
        print ("Your portfolio includes: {} $ cash,\n{} amount of stocks,\n{} shares of mutual funds.\n{} shares of bonds.".format(self.cash, self.stock, self.fund, self.bond))

    def history(self):
        print("\nHere is the resume of your transactions:")
        for i in self.resume:
            print(i)           
    
portfolio = Portfolio() 
portfolio.addCash(300.50)
s = Stock(20, "HFH") 
portfolio.buyStock(5, s)
portfolio.withdrawCash(50)
portfolio.sellStock(1,s)
mf1 = MutualFund("BRT")
mf2 = MutualFund("GHT") 
b1=Bond(2,"bnd")
portfolio.buyBond(3,b1)
portfolio.buyMutualFund(10.3, mf1) 
portfolio.buyMutualFund(2, mf2)
portfolio.sellMutualFund(3,mf1)
portfolio.print_portfolio()
portfolio.history()