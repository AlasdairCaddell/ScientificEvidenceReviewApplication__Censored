#from enum import Enum

##Type def
#0
#
#
#
#
#
#


class HeadNode:
    def __init__(self,data):
        self.child=None
        self.data=data #dict containing author/institution,cover,header,footer
        
        
    def setChild(self,child):
        self.child=child
        
        
class SectionNode:
    def __init__(self,data,section,text):
        self.child=None
        self.sibling=None
        self.data=data # dict with section name, location
        self.section=section
        self.text=text
        
    def setChild(self,child):
        self.child=child
        
        
        
        
class MeaningNode:
    def __init__(self,data):
        self.child=None
        self.sibling=None
        
    def setChild(self,child):
        self.child=child
        
        
class ReferenceNode:

    def __init__(self,data):
        self.child=None
        self.sibling=None
        self.data=data
        
    def refer(self,data):
        
        n,d=data.pop()
        
        sib=MeaningNode
        self.setSibling(sib)
        sib.refer(data)
        
    def setChild(self,child):
        self.child=child
        
    def setSibling(self,sibling):
        self.sibling=sibling
        