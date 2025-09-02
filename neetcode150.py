import math
from collections import defaultdict
class listnode:
    def __init__(self,value=0,next=None):
        self.value=value
        self.next=next

class Minstack:
    def __init__(self):
        self.stack=[]
        self.minstack=[]

    def push(self,val):
        self.stack.append(val)
        if self.minstack:
            val=min(val,self.minstack[-1])
        self.minstack.append(val)

    def pop(self):
        self.stack.pop()
        self.minstack.pop()

    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.minstack[-1]


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
    closetoopen={")":"(","}":"{","]":"["}

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


def lengthOfLongestSubstring(s):
    mp={}
    l=0
    res=0
    for r in range(len(s)):
        if s[r] in mp:
            l=max(l,mp[s[r]]+1)
        mp[s[r]]=r
        res=max(res,r-l+1)
    return res


def maxDepth(root):
    if root==None:
        return 0
    else:
        value1=self.maxDepth(root.left)+1
        value2=self.maxDepth(root.right)+1
        if value1>=value2:
            return value1
        else:
            return value2


def searchMatrix(matrix,target):
        l=0
        r=len(matrix)-1
        array=[]
        while l<=r:
            middle=(l+r)//2
            if matrix[middle][0]>target:
                r=middle-1
            elif matrix[middle][-1]<target:
                l=middle+1
            else:
                array=matrix[middle]
                break
        if not array:
            return False
        l=0
        r=len(array)-1
        while l<=r:
            middle=(l+r)//2
            if array[middle]>target:
                r=middle-1
            elif array[middle]<target:
                l=middle+1
            else:
                return True

        return False


def subsets(nums):
    res=[]
    subset=[]
    # for num in nums:
    #     res+=[subset+[num] for subset in res]
    # return res  
    def dfs(i):#dfs
        if i>=len(nums):
            res.append(subset.copy())
            return
        subset.append(nums[i])
        dfs(i+1)
        subset.pop()
        dfs(i+1)
    dfs(0)
    return res


def combinationSum(nums,target):
    res=[]
    def dfs(i,cur,total):
        if total==target:
            res.append(cur.copy())
            return
        elif i>=len(nums) or total>target:
            return
        cur.append(nums[i])
        dfs(i,cur,total+nums[i])
        cur.pop()
        dfs(i+1,cur,total)
    dfs(0,[],0)
    return res


def productExceptSelf(nums):
    res=[1]*len(nums)
    prefix=1
    for i in range(len(nums)):
        res[i]=prefix
        prefix*=nums[i]
    postfix=1
    for i in range(len(nums)-1,-1,-1):
        res[i]*=postfix
        postfix*=nums[i]
    return res


def threesum(nums):
    indexes=[]
    res=[]
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            res.append([nums[i],nums[j]])
            indexes.append([i,j])
    #print(indexes)
    temp=set()
    for i in range(len(res)):
        for j in range(len(nums)):
            if j not in indexes[i]:
                #print(j)
                total=nums[j]
                res[i].append(nums[j])
                #print(res[count])
                for k in range(2):
                    total+=res[i][k]
                if total==0:
                    res[i].sort()
                    temp.add(tuple(res[i]))
            res[i]=res[i][0:2]
    res=[]
    for i in temp:
        res.append(list(i))
    return res


def evalRPN(tokens):
    # nums=[]
    # answer=0
    # first=True
    # for i in range(len(tokens)):
    #     if tokens[i].isdigit() or (tokens[i][0]=='-' and len(tokens[i])>1):
    #         a=1
    #         if tokens[i][0]=='-':
    #             a=-1
    #             tokens[i]=tokens[i][1:]
    #         nums.append(int(tokens[i])*a)
    #         if first:
    #             answer=int(tokens[i])*a
    #             first=False
    #     else:
    #         top=nums[-1]
    #         second=nums[-2]
    #         nums.pop()
    #         nums.pop()
    #         if tokens[i]=="+":
    #             answer=second+top
    #         elif tokens[i]=="-":
    #             answer=second-top
    #         elif tokens[i]=="*":
    #             answer=second*top
    #         elif tokens[i]=="/":
    #             answer=int(second/top)
    #         nums.append(answer)
    # return answer
    stack = []#also no need for answer
    for c in tokens:
        if c == "+":#optimized code where YOU know what the else statement is going to be so you can avoid what i did above (convert positive and integer strings to int #not good)
            stack.append(stack.pop() + stack.pop())
        elif c == "-":
            a, b = stack.pop(), stack.pop()
            stack.append(b - a)
        elif c == "*":
            stack.append(stack.pop() * stack.pop())
        elif c == "/":
            a, b = stack.pop(), stack.pop()
            stack.append(int(float(b) / a))
        else:
            stack.append(int(c))
    return stack[0]#just have to return the only element in the list


def minEatingSpeed(piles,h):
    piles.sort()
    smallest=1
    biggest=piles[-1]
    res=biggest
    while smallest<=biggest:#this makes sense because we got all the integers we needed, we don't need to use integers outside of these bounds
        hoursp=0
        middle=(smallest+biggest)//2
        for i in range(len(piles)):
            hoursp+= math.ceil(float(piles[i]) / middle)
        if hoursp>h:#hours passed is more than hours means we need to make middle be bigger so that hoursp can be smaller, doesn't meet the requirement 
            smallest=middle+1
        else:
            res=middle#in this else statement hoursp is either equal to hours or less than hours so we can record middle to res since it satisfies the time constraint
            biggest=middle-1
    return res


def isvalidSudoku(board):
    rows=defaultdict(set)
    cols=defaultdict(set)
    squares=defaultdict(set)
    for r in range(9):
        for c in range(9):
            if board[r][c].isdigit():
                if board[r][c] in rows[r] or board[r][c] in cols[c] or board[r][c] in squares[(r//3,c//3)]:
                    return False
                rows[r].add(board[r][c])
                cols[c].add(board[r][c])
                squares[(r//3,c//3)].add(board[r][c])
    return True

def reorderList(head):
    slow=head
    fast=head.next
    while fast and fast.next:
        slow=slow.next
        fast=fast.next.next
    
    second=slow.next
    prev=slow.next=None
    while second:
        temp=second.next
        second.next=prev
        prev=second
        second=temp
    first=head
    second=prev
    while second:
        temp1,temp2=first.next,second.next
        first.next=second
        second.next=temp1
        first=temp1
        second=temp2

def checkInclusion(s1,s2):
    count1=defaultdict(int)
    for i in range(len(s1)):
        count1[s1[i]]+=1
    for i in range((len(s2)+1)-len(s1)):
        if s2[i] in count1:
            count2=defaultdict(int)
            for j in range(len(s1)):
                count2[s2[i+j]]+=1
            if count2==count1:
                return True
    return False


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
    # head=listnode(1)
    # l1=listnode(2)
    # l2=listnode(3)
    # l3=listnode(4)
    # head.next=l1
    # l1.next=l2
    # l2.next=l3
    # l3.next=l2
    # print(hasCycle(head))
    # s=encode(["neet","code","love","you"])
    # print(decode(s))
    #print(subsets([1,2,3]))
    print(threesum([-1,0,1,2,-1,-4]))
main()