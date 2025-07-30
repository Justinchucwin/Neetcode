class listnode:
    def __init__(self,value=0,next=None):
        self.value=value
        self.next=next

def binarysearch(array,l,r,target):
    if r<l:
        return -1
    else:
        mid=(l+r)//2
        if array[mid]==target:
            return mid
        elif array[mid]>target:
            return binarysearch(array,l,mid-1,target)
        else:
            return binarysearch(array,mid+1,r,target)


def maxProfit(prices):#index is day, elements are price
    l,r=0,1
    maxp=0

    while r<len(prices):
        if prices[l]<prices[r]:
            profit=prices[r]-prices[l]
            maxp=max(maxp,profit)
        else:
            l=r
        r+=1
    return maxp


def isAnagram(s,t):
    counts,countt={},{}
    if len(s)!=len(t):
        return False
    for i in range(len(s)):
        counts[s[i]]=1+counts.get(s[i],0)
        countt[t[i]]=1+countt.get(t[i],0)
    return counts==countt


def scoreofstring(s):
    sum=0
    lengthstr=len(s)
    for i in range(lengthstr//2):
        sum+=abs(ord(s[i])-ord(s[i+1]))
        if lengthstr%2==0 and (lengthstr//2)==i:
            break
        endpos=lengthstr-1-i 
        sum+=abs(ord(s[endpos])-ord(s[endpos-1]))
    return sum


def validpalindrome(s):
    startind=0
    endind=len(s)-1
    while endind>startind:
        if s[startind].lower()==s[endind].lower():
            startind+=1
            endind-=1
        else:
            if s[startind].isalnum() and s[endind].isalnum():
                return False
            else:
                if not s[startind].isalnum():
                    startind+=1
                if not s[endind].isalnum():
                    endind-=1     
    return True


def isValid(s):#gets a string as an arguement and see's if it follows rule of closed parentheses
    stack=[]
    closetoopen={")":"(","}":"{","[":"]"}

    for i in s:
        if i in closetoopen:
            if stack and closetoopen[i]==stack[-1]:#means stack is not empty
                stack.pop()
            else:
                return False
        else:
            stack.append(i)
    if not stack:
        return True
    return False


def reverselist(head):
    prev,curr=None,head

    while curr:
        temp=curr.next
        curr.next=prev
        prev=curr
        curr=temp
    return prev
            

            


def main():
    #print(binarysearch([1,2,3,4,5,6],0,5,4))
    #print(maxProfit([10,8,7,5,2]))
    #print(isAnagram("racecar","carrace"))
    head=listnode(0)
    current=head
    for i in range(3):
        ln=listnode(i+1)
        current.next=ln
        current=current.next

       


main()