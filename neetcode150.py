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


def main():
    #print(binarysearch([1,2,3,4,5,6],0,5,4))
    #print(maxProfit([10,8,7,5,2]))
    print(isAnagram("racecar","carrace"))


main()