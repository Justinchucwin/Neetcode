import math
from collections import defaultdict
class listnode:
    def __init__(self,value=0,next=None):
        self.value=value
        self.next=next

class Minstack:#if we want to keep track of the minimum value we need a second stack because we need to keep track of when the min value was added because if it's popped then we need to rely on a new minimum value which was added prior to the recent one being popped
    def __init__(self):
        self.stack=[]
        self.minstack=[]

    def push(self,val):
        self.stack.append(val)
        if self.minstack:
            val=min(val,self.minstack[-1])#prev min value could still be top. top should always be min val and a min val should always be added each push to keep track of where the real on is 
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
    

#if we're trying to the kth largest element in the array, we can just sort the array each time we add it and then get the nums[-k]
class KthLargest:
    def __init__(self,k,nums):
        self.k=k
        self.nums=nums
        
    def add(self,val):
        self.nums.append(val)
        self.nums.sort()#sort() changes the list, while sorted makes a copy
        return self.nums[-self.k]


def invertTree(root):#for each iteration in recursion we want to switch the left child and the right child save the left child and then swap. 
    if root==None:
        return
    temproot=root.left
    root.left=root.right
    root.right=temproot
    invertTree(root.left)
    invertTree(root.right)
    return root



def binarysearch(array,l,r,target):#time complexity O(log(n)) splits the array in half each time based off of middle and whether target is bigger or smaller than that. 
    if r<l:#for recursive it's not <= but just less than cause there could be a case of only having one element 
        return -1
    else:
        mid=(l+r)//2
        if array[mid]==target:
            return mid
        #if array index mid is target then we just return the index
        #if not then we can do this recurisvely where if target is bigger than the current position we use the latter half of the list and vice versa(latter half is l=middle+1 to r) amd vice versa 
        elif array[mid]>target:
            return binarysearch(array,l,mid-1,target)
        else:
            return binarysearch(array,mid+1,r,target)


def maxProfit(prices):#index is day, elements are price. for this one if we find a value that's lower than the previous we just use that value instead and compare it to the later values because lower buy is better than higher buy and we can just keep track of maxprofits
    l,r=0,1
    maxp=0

    while r<len(prices):#we've checked the whole list already if r reaches the last element
        if prices[l]<prices[r]:#seeing if the profit we gained here is bigger than the prev
            profit=prices[r]-prices[l]
            maxp=max(maxp,profit)
        else:
            l=r
        r+=1
    return maxp


def isAnagram(s,t):#anagrams will have the same count of letters, no better way to do this than to use a hashmap. Unique letters(key) with how many times they repeat(value) can't use set because if there's a repeat character we'll never know with a set and then we can't compare words
    counts,countt={},{}
    if len(s)!=len(t):
        return False
    for i in range(len(s)):
        counts[s[i]]=1+counts.get(s[i],0)#dict.get() makes it so that it gets a value and can also initialize a value for a newly created key
        countt[t[i]]=1+countt.get(t[i],0)
    return counts==countt


def scoreofstring(s):#if i were to redo this i would iterate from (1,len(s)) and have it so that it takes the abs difference of prev and current then add it to the total. Below has a better time complexity O(n/2) but the way i just described it sounds easier
    sum=0
    lengthstr=len(s)
    for i in range(lengthstr//2):
        sum+=abs(ord(s[i])-ord(s[i+1]))#ord gives ascii value
        if lengthstr%2==0 and (lengthstr//2)==i:
            break
        endpos=lengthstr-1-i 
        sum+=abs(ord(s[endpos])-ord(s[endpos-1]))
    return sum


def validpalindrome(s):#there can be spaces in the string so letter a certain amount of position away from the beginning compared to the mirror could be different if left and right pointer equal each otherthen it's a palindrome but if it doesn't reach there two characters don't match up
    startind=0
    endind=len(s)-1
    while endind>startind:
        if s[startind].lower()==s[endind].lower():
            startind+=1
            endind-=1
        else:
            if s[startind].isalnum() and s[endind].isalnum():#isalnum() is alpha numeric 
                return False
            else:
                if not s[startind].isalnum():
                    startind+=1
                if not s[endind].isalnum():
                    endind-=1     
    return True


def isValid(s):#gets a string as an arguement and see's if it follows rule of closed parentheses. pushes open parenthesis in and if it encounters a close parenthesis checks the top of the stack to see if it matches if stack is empty by the end it returns true cause it popped out all the open parenthesis
    stack=[]
    closetoopen={")":"(","}":"{","]":"["}

    for i in s:
        if i in closetoopen:
            if stack and closetoopen[i]==stack[-1]:#if list checks to see whether the list is empty or not 
                stack.pop()
            else:
                return False
        else:
            stack.append(i)
    if not stack:
        return True
    return False


def reverselist(head):#in order to reverse you have to initialize prev if the first node is going to be the last then the last will point to none therfore prev has to be none. save the curr.next set curr.next=prev than prev to current then curr to temp
    prev,curr=None,head

    while curr:
        temp=curr.next
        curr.next=prev
        prev=curr
        curr=temp
    return prev


def hasCycle(head):#i stored all nodes in a hashmap and if the current node is in the hashmap than there's a cycle. we could just use set. We could also do two pointers so that the space complexity is O(1) if there's a cycle it's infinitely repeating so slow and fast will eventually have to meet up
    curr=head
    count={}
    while curr!=None:
        count[curr]=1+count.get(curr,0)
        if count[curr]>1:
            return True
        curr=curr.next
    return False


def mergeTwoLists(head1,head2):#solution makes it easy, while there are still more nodes compare the nodes. add the lesser node to node then iterate once through the lesser node list. at the end add the remainder to the node
    # current1=head1
    # current2=head2
    # if head1.value<=head2.value:
    #     current3=linklist(head1.value)
    #     head3=listnode(current1.value)
    #     current1=current1.next
    # else:
    #     current3=linklist(head2.value)
    #     head3=listnode(current2.value)
    #     current2=current2.next
    # current3=head3
    # while current1!=None and current2!=None:
    #     if current1.value<=current2.value:
    #         current3.next=listnode(current1.value)
    #         current1=current1.next
    #     else:
    #         current3.next=listnode(current2.value)
    #         current2=current2.next
    #     current3=current3.next
    # if current1==None:
    #     current3.next=current2
    # else:
    #     current3.next=current1
    # return head3
    dummy = node = listnode()#we make dummy so we can have a reference to the beggining of the merged list, and node to create the mergedlist

    while list1 and list2:
        if list1.val < list2.val:
            node.next = list1
            list1 = list1.next
        else:
            node.next = list2
            list2 = list2.next
        node = node.next

    node.next = list1 or list2#add the remaining onto the merged list
    return dummy.next#dummy next is the start since the current is just an empty node pointing to the start

            

def climbStairs(n):#for this i imagine that there are two ways to get to the second to last and third to last step making it so that there has to be at least 3 steps making the base case n<=2 and the rest being the sum of the total combinations of second to last and third to last steps
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


def canAttendMeeting(intervals):#sort the meetings by start time, and if the end of the meeting is greater than the next elements start it will clash. 
    sortedintervals=intervals.sort(key=lambda x:x.start)#lambda is a mini function that returns x.start for all the x's and the key in sort sorts it by x.start
    for i in range(1,len(intervals)):
        if intervals[i-1].end>intervals[i].start:
            return False
    return True


#purpose of this functions is to take a list of strings and then append sublists of anagrams into a new list. we intialize a dictionary where every key is a list. 
#Each list is initialized to 26 zeroes and the string makes up a unique list identifier by changing the zero to the count of how many letters are in the string. then stores the key(unique identifier) and value(list of words)
#then if a word with the same key pops up it appends that word to the value
def groupAnagrams(strs):
    # counts=[]
    # for i in range(len(strs)):
    #     count={}
    #     for j in range(len(strs[i])):
    #         count[strs[i][j]]=1+count.get(strs[i][j],0)
    #     counts.append(count)
    # categorized=[]
    # for i in range(0,len(counts)-1):
    #     if counts[i]!=-1:
    #         categorized.append([strs[i]])
    #         for j in range(i+1,len(counts)):
    #             if counts[i]==counts[j]:
    #                 categorized[-1].append(strs[j])
    #                 counts[j]=-1
    # if counts[-1]!=-1:
    #     categorized.append([strs[-1]])
    # return categorized
    res = defaultdict(list)#list represents the value type
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1#uses ord to see where the letter increments within count
        res[tuple(count)].append(s)#have to change count to tuple because can't have lists as key because it can't be mutable, appends the word that is an anagram
    return list(res.values())#turns the dictionary object into a list 



def hasDuplicate(nums):#sort makes it so that duplicates are together so that you can just check prev to next, however if i were to do this again i would just make a set and add the elements within the set until there's a duplicate then return true
    nums.sort()
    for i in range(1,len(nums)):
        if nums[i-1]==nums[i]:
            return True
    return False


#both have O(n) time complexity, with this we need to see if target-current is within the list. So we need hash map where key(element) and value(index)
#once we did that, we can start from the start of the list calculate the difference(target-current) see if it's in the hash map and if it is and isn't the same indexes return the list where smallest index is first
def twoSum(nums,target):
    l,r=0,len(nums)-1#two pointers, space complexity O(1) this method is different where if it's sorted you can just add 1 to l if current sum of pointers is less than target or minus 1 to r if current sum of pointers is greater than target return when you reach target
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


#when i think of frequency i think of hashmap where key(element) value(frequency)
#If we want to the kth-1st most frequent numbers gotta sort the dictionary. 
#the only way to sort dictionaries are changing it into a list of tuples
#sorting the list then changing it back to a dictionary then getting the keys then converting it into a list so you can get the kth-1st most frequent numbers
def topKFrequent(nums,k):
    counts={}
    for i in nums:
        counts[i]=1+counts.get(i,0)
    keysorted=list(dict(sorted(counts.items(),key=lambda x:x[1])).keys())#.items() turns a dictionary into an dict object that is a list of tuples, dict( changes it back to a dict, and keys changes a dict to a dict object that's a list with the keys, then list just converts it to list
    return keysorted[-k:]



# def encode(strs):
#     return ",".join(strs)


# def decode(s):
#     start=0
#     decoded=[]
#     if s=="":
#         return [""]
#     for i in range(len(s)):
#         if s[i]=='#':
#             decoded.append(s[start:i])
#             start=i+1
#         if i==len(s)-1:
#             decoded.append(s[start:])
#     return decoded



#in order to decode. The only way is to encode by putting numbers followed by a pound followed by the string.
#the number is used to determine when the string ends so it can begin using the number for the length on the next string
#and pound is used to decipher when the number is done
#two pointer is used one for the start of the number and one that iterates towards the end of the number,
#once it hits pound then you set the start of the pounter right after pound and j to i+ length to get the string
#then append
def encode(strs):
    res = ""
    for s in strs:
        res += str(len(s)) + "#" + s
    return res


def decode(s):
    res = []
    i = 0

    while i < len(s):
        j = i
        while s[j] != '#':
            j += 1
        length = int(s[i:j])
        i = j + 1
        j = i + length
        res.append(s[i:j])
        i = j

    return res


#substring in this case is something that doesn't have a repeating character
#once it sees something with a repeating character it then removes all the char key in the dictionary from the start of the pointer to after it repeats
#then the cycle starts over. res is comparing previous res to r-l+1 so it can return the maxres
def lengthOfLongestSubstring(s):
    # mp={}
    # l=0
    # res=0
    # for r in range(len(s)):
    #     if s[r] in mp:
    #         l=max(l,mp[s[r]]+1)
    #     mp[s[r]]=r
    #     res=max(res,r-l+1)
    # return res
    charSet = set()
    l = 0
    res = 0

    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
            l += 1
        charSet.add(s[r])
        res = max(res, r - l + 1)
    return res



#if it goes through 1 node then it will return the value of the depth plus 1(1 being the current node)
#value keeps on getting added for the current node
#then once it reaches back to the ture it will compare the values
#then that will be max depth
#when you see tree think of recursion
def maxDepth(root):
    if root==None:
        return 0
    else:
        value1=maxDepth(root.left)+1
        value2=maxDepth(root.right)+1
        if value1>=value2:
            return value1
        else:
            return value2


#if matrix is sorted 2d then binary search the first dimension
#if there is no array that contains number, return false
#but if there is then binary search through the second dimension
def searchMatrix(matrix,target):
        l=0
        r=len(matrix)-1
        array=[]
        while l<=r:#for iteration l should equal r this time because it checks the condition before entering the loop
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


#when you see all possible configurations, like subsets think backtracking 
def subsets(nums):
    res=[]#res to append subsets
    subset=[]#subsets to store the numbers
    # for num in nums:
    #     res+=[subset+[num] for subset in res]
    # return res  
    def dfs(i):#dfs
        if i>=len(nums):
            res.append(subset.copy())
            return
        subset.append(nums[i])
        dfs(i+1)#move onto the next number

        #this part is the backtracking part it pops then incorporates a new element
        subset.pop()
        dfs(i+1)
    dfs(0)
    return res


#same concept as above but we're instead trying to reach total==target. But if we pass i<len(nums) or total>target then we have to return also. Also since we're using repeatable elements one of the recursive calls we don't change i
def combinationSum(nums,target):
    res=[]
    def dfs(i,cur,total):
        if total==target:
            res.append(cur.copy())
            return
        elif i>=len(nums) or total>target:
            return
        cur.append(nums[i])
        dfs(i,cur,total+nums[i])#this line is so we can use the same element
        cur.pop()
        dfs(i+1,cur,total)#this line is for using a different element, we're not incrementing nums[i] to total because it just got popped
    dfs(0,[],0)
    return res



#for this one we're tryign to get two arrays where prefix is the products of the elements from left to right and postfix is the products of elements from right to left, and then for each to multiply it with each other
#this would result in the product for each i to be everything but itself
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


#reverse polish notation requires a stack because we're using the most recent element we pushed onto the stack.
#if using most recent elements we pushed in (last in first out)
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
    #iterating through list and if it's anything besides operators then it's a number and you just append it onto a list
    #if it is an operator then you take out the two top elements then do the math accordingly then add the answer back in
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


#so for this one we know that we're trying to find the min eating speed (bannanas per hour)
#we know that the it's in between 1 and the max value in the array piles
#1 being the lowest positive integer and the max value in the array piles because we want the minimum eating speed(we can go larger but we want to find minimum and the max in the array is the highest the minimum CAN possible be)
#meaning that we use binary search if we're searching for a numbers 1-n
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


#for this one, we're checking unique values for the rows, the columns, and the squares meaning we should have three dictionaries where the values are sets since each respective rows, columns, and squares have their unique values in them
#For the keys for reach, rows is the row number, col is the col number, and sqaures keys is represented by a unique identifier for the square
#if the square is within it's 3x3 then it's a valid square and we determine the square by r//3 and c//3 for its key
#if a set already has the current element return false because it violates sudoku
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


#for reorder list we need to recognize that the latter half of the link list will be the evens and the first half of the list will be the odds
#meaning we need to obtain the latter half of the list
#slow and fast will give you the latter half of the list because slow will reach the halfway point when fast reaches the end
#then from there reverse the latter half of the list because we want the odds to be reversed
#then zipper merge the even with odds
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



#It said permutation of a string in this problem so i think of hash map where key(character) value(frequency)
#get the frequencies for the first string so that we can compare it to the frequencies for the substring within string 2 
def checkInclusion(s1,s2):
    count1=defaultdict(int)
    for i in range(len(s1)):
        count1[s1[i]]+=1
    for i in range((len(s2)+1)-len(s1)):
        if s2[i] in count1:#realizes that one of the letters is within the hashmap
            count2=defaultdict(int)
            for j in range(len(s1)):#evaluates the frequencies for the possible string permutation
                count2[s2[i+j]]+=1
            if count2==count1:#checks to see if frequencies match up
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