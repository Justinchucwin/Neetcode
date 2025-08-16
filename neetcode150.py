class listnode:
    def __init__(self,value=0,next=None):
        self.value=value
        self.next=next



class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



class Interval:
    def __init__(self,start,end):
        self.start=start
        self.end=end



class linklist:
    def __init__(self,head):
        self.head=head

    def display(self):
        current=self.head
        while current.next!=None:
            print(current.value,end=",")
            current=current.next
        print(current.value)

    def add(self,value):
        current=self.head
        while current.next!=None:
            current=current.next
        current.next=listnode(value)
    


class KthLargest:
    def __init__(self,k,nums):
        self.k=k
        self.nums=nums
        
    def add(self,val):
        self.nums.append(val)
        self.nums.sort()
        return self.nums[-self.k]


def invertTree(root):
    if root==None:
        return
    temproot=root.left
    root.left=root.right
    root.right=temproot
    invertTree(root.left)
    invertTree(root.right)
    return root



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
            
def hasCycle(head):
    curr=head
    count={}
    while curr!=None:
        count[curr]=1+count.get(curr,0)
        if count[curr]>1:
            return True
        curr=curr.next
    return False

def mergeTwoLists(head1,head2):
    current1=head1
    current2=head2
    if head1.value<=head2.value:
        current3=linklist(head1.value)
        head3=listnode(current1.value)
        current1=current1.next
    else:
        current3=linklist(head2.value)
        head3=listnode(current2.value)
        current2=current2.next
    current3=head3
    while current1!=None and current2!=None:
        if current1.value<=current2.value:
            current3.next=listnode(current1.value)
            current1=current1.next
        else:
            current3.next=listnode(current2.value)
            current2=current2.next
        current3=current3.next
    if current1==None:
        current3.next=current2
    else:
        current3.next=current1
    return head3
            

def climbStairs(n):
    # def dfs(i): recursive way O(2^n)
    #     if i>=n:
    #         return i==n
    #     return dfs(i+1)+dfs(i+2)
    if n<=2:#bottom up dynamic O(n)
        return n
    ways=[0]*(n+1)
    ways[1],ways[2]=1,2
    for i in range(3,n+1):
        ways[i]=ways[i-1]+ways[i-2]
    return ways[i]
    #return dfs(0)


def canAttendMeeting(intervals):
    sortedintervals=intervals.sort(key=lambda x:x.start)
    for i in range(1,len(intervals)):
        if intervals[i-1].end>intervals[i].start:
            return False
    return True


def groupAnagrams(strs):
    counts=[]
    for i in range(len(strs)):
        count={}
        for j in range(len(strs[i])):
            count[strs[i][j]]=1+count.get(strs[i][j],0)
        counts.append(count)
    categorized=[]
    for i in range(0,len(counts)-1):
        if counts[i]!=-1:
            categorized.append([strs[i]])
            for j in range(i+1,len(counts)):
                if counts[i]==counts[j]:
                    categorized[-1].append(strs[j])
                    counts[j]=-1
    if counts[-1]!=-1:
        categorized.append([strs[-1]])
    return categorized


def hasDuplicate(nums):
    nums.sort()
    for i in range(1,len(nums)):
        if nums[i-1]==nums[i]:
            return True
    return False


def twoSum(nums,target):#both have O(n) time complexity
    l,r=0,len(nums)-1#two pointers, space complexity O(1)
    while l<r:
        cursum=nums[l]+nums[r]
        if  cursum<target:
            l+=1
        elif cursum>target:
            r-=1
        else:
            return [l+1,r+1]
    return []
    # indices={}#hash map, space complexity O(n)
    # for i,n in enumerate(nums):
    #     indices[n]=i
    # for i,n in enumerate(nums):
    #     diff=target-n
    #     if diff in indices and i!=indices[diff]:
    #         return [i,indices[diff]]


def topKFrequent(nums,k):
    counts={}
    for i in nums:
        counts[i]=1+counts.get(i,0)
    keysorted=list(dict(sorted(counts.items(),key=lambda x:x[1])).keys())#.items() turns a dictionary into an dict object that is a list of tuples, dict( changes it back to a dict, and keys changes a dict to a dict object that's a list with the keys, then list just converts it to list
    return keysorted[-k:]

def encode(strs):
    return ",".join(strs)

def decode(s):
    start=0
    decoded=[]
    if s=="":
        return [""]
    for i in range(len(s)):
        if s[i]=='#':
            decoded.append(s[start:i])
            start=i+1
        if i==len(s)-1:
            decoded.append(s[start:])
    return decoded



def main():
    #print(binarysearch([1,2,3,4,5,6],0,5,4))
    #print(maxProfit([10,8,7,5,2]))
    #print(isAnagram("racecar","carrace"))
    # list1=linklist(listnode(1))
    # list1.add(2)
    # list1.add(4)
    # list2=linklist(listnode(1))
    # list2.add(3)
    # list2.add(5)
    # list3=linklist(mergeTwoLists(list1.head,list2.head))
    # list3.display()
    # k=KthLargest(3, [1, 2, 3, 3])
    # print(k.add(3))
    # print(k.add(5))
    # print(k.add(6))
    # print(k.add(7))
    # print(k.add(8))
    #print(groupAnagrams(["act","pots","tops","cat","stop","hat"]))
    head=listnode(1)
    l1=listnode(2)
    l2=listnode(3)
    l3=listnode(4)
    head.next=l1
    l1.next=l2
    l2.next=l3
    l3.next=l2
    print(hasCycle(head))
    # s=encode(["neet","code","love","you"])
    # print(decode(s))
main()