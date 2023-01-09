from math import gcd
from copy import deepcopy
def Gongyueshu():
    n = 4
    i = 2
    List = []
    lista = []
    if n == 0 or n == 1:
        return List
    while i <= n:
        for j in range(1, i):
            if j/i not in lista:
                num = str(j) + '/' + str(i)
                num = str(num)
                List.append(num)
                lista.append(j/i)
            j+=1
        i+=1
    print(List)
def test(n):
    List = []
    for i in range(2, n + 1):
        for j in range(1, i):
            if gcd(i, j) == 1:
                List.append(f"{j}/{i}")
    return List
    # return [f"{j}/{i}" for i in range(2, n + 1) for j in range(1, i) if gcd(i, j) == 1]
def zuixiao1848(nums,t,s):
    m = 999
    for i in range(0, len(nums)):
        if nums[i] == t:
            m = min(m, abs(i - s))
    return m
def jvzehn867():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    m, n = len(matrix), len(matrix[0])
    L = [[0] * m for _ in range(n)]
    for i in range(0,m):
        for j in range(0,n):
            L[j][i] = matrix[i][j]
    print(L)
def yichu27(nums):
    val = 3
    nums = [n for n in nums if n != val]
    print(nums)
def yihuo2433():
    pref = [5, 2, 0, 3, 1]
    L = []
    k = 0
    for i in range(0,len(pref)):
        if L:
            k = k ^ L[-1]
        L.append(k^pref[i])
    print(L)
def yuanzu1726():
    import itertools
    nums = [1,2,4,8,16,32,64,128]
    i=0
    for a,b,c,d in itertools.combinations(nums, 4):
        if a*b == c*d or a*c == b*d or a*d ==b*c:
            i+=1
            print(a,b,c,d)
    print(i*8)
from collections import defaultdict
def tupleSameProduct(nums):
    n = len(nums)
    res = 0
    cnt = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            # print(cnt[nums[i] * nums[j]])
            res += cnt[nums[i] * nums[j]]
            cnt[nums[i] * nums[j]] += 1
            print(cnt)
    return res * 8
def zidianxu440():
    s = ['1','10','7','8','11','123']
    print(sorted(s))
def removeDuplicates1047(s) :
    s = list(s)
    ii = 0
    def p():
        if  not s:
            return 1
        for i in range(0,len(s)):
            if i ==len(s) - 1:
                return 1
            else:
                if s[i] == s[i+1]:
                    s[:] = s[:i]+s[i+2:]
                    break
        return 0
    while ii==0:
        ii = p()
        print(ii)
    s = ''.join(s)
    return s
def remove1047():
    s = "cabbac"
    s = list(s)
    a = []
    for i in range(0,len(s)):
        if not a:
            a.append(s[i])
        else:
            if s[i] == a[-1]:
                a.pop(-1)
            else:
                a.append(s[i])
    s = ''.join(a)
    return s
def zishuzu907():
    arr = [3, 1, 2, 4]
    stack= []
    Sum = sum(arr)
    for i in range(0,len(arr)):
        stack2 = []
        for j in range(i,len(arr)):
            if i ==0:
                stack.append(arr[j])
            elif arr[j] < stack[j-i]:
                stack2.append(arr[j])
            else:
                stack2.append(stack[j-i])
        if stack2:
            Sum = Sum + sum(stack2)
            stack = deepcopy(stack2)
    print(Sum)
def shenji481():
    s1 = [1,2,2]
    s2 = [1,2]
    n = 10
    while len(s1) < n+1:
        s2.append(s1[len(s2)])
        if s2[-1] == 1:
            if s1[-1] == 1:
                s1.append(2)
            else:
                s1.append(1)
        if s2[-1] == 2:
            if s1[-1] == 1:
                s1.append(2)
                s1.append(2)
            else:
                s1.append(1)
                s1.append(1)
    print(s1)
def zifuchuan1662():
    w1 = ['ab','c']
    w2 = ['a','bc']
    # s1= ''
    # s2= ''
    # for i,j in w1,w2:
    #     print(i,j)
    #     s1 = s1+i
    #     s2 = s2+j
    # print(s1,s2)
    return ''.join(w1) == ''.join(w2)
def xinhaota1620():
    towers = [[44,31,4],[47,27,27],[7,13,0],[13,21,20],[50,34,18],[47,44,28]]
    radius = 13
    max = [0, 0, 0]
    for i in range(50):
        for j in range(50):
            sum = 0
            for k in towers:
                if ((k[0]-i)**2 + (k[1]-j)**2)**0.5 <= radius:
                    sum = sum + int (k[2]/ (1 + ((k[0]-i)**2 + (k[1]-j)**2)**0.5))
            print(i,j,sum)
            if sum > max[2]:
                max=[i,j,sum]
    print(max)
def zuidazfc1668():
    sequence = "aaabaaaabaaabaaaabaaaabaaaabaaaaba"
    word = "aaaba"
    a = word
    i = 0
    while word in sequence:
        i +=1
        word = word + a
    print(i)
def shuzi754(target):
    t = abs(target)
    sum = 0
    i = 0
    while sum < t:
        i += 1
        sum = sum + i
    if sum == t:
        return i
    elif (sum - t) % 2 == 0:
        return i
    else:
        if i % 2 == 0:
            return i + 1
        else:
            return i + 2
def buer1106():
    expression = "|(&(t,f,t),!(t))"
    stk = []
    for i in expression:
        if i == ',':
            continue
        if i != ')':
            stk.append(i)
            continue
        t = f = 0
        while stk[-1] != '(':
            if stk.pop() == 't':
                t += 1
            else:
                f += 1
        stk.pop()
        op = stk.pop()
        if op == '!':
            stk.append('t' if f==1 else 'f')
        elif op =='&':
            stk.append('t' if f==0 else 'f')
        elif op == '|':
            stk.append('t' if t else 'f')
    return (True if stk[-1] == 't' else False)
def Goal1678():
    pass
def shibai816():
    s = "(001234)"
    t = []
    tt = []
    for j in  range(1,len(s)-2):
        a = ''
        for i in range(0,len(s)):
            if s[i] != '(' and s[i] != ')':
                if len(a)==j:
                    a = a + ','
                a = a +s[i]
        t.append(a)
    for b in t:
        y,x = b.split(',')
        print(y,x)
        if len(x) >1:
            for j in range(1,len(x)):
                a = ''
                for i in range(0,len(x)):
                    if len(a) == j:
                        a = a +'.'
                    a = a + x[i]
                tt.append('('+y+','+a+')')
        if len(y) >1:
            for j in range(1,len(y)):
                b = ''
                for i in range(0,len(y)):
                    if len(b) == j:
                        b = b +'.'
                    b = b + y[i]
                tt.append('(' + b + ',' + x + ')')

    print(t)
    print(tt)
def zuidajiahao764():
    n = 1
    mines = [[0,0]]
    for jie in range(0,round((n+1)/2)):
        flag = 0
        for i in range(0+jie,n-jie):
            for j in range(0+jie,n-jie):
                flag = 0
                for t in range(0,jie+1):
                    if [i + t, j]  not in mines:
                        if [i - t, j] not in mines:
                            if [i ,j + t] not in mines:
                                if [i ,j - t] not in mines:
                                    flag += 1
                if flag == jie +1:
                    break
            if flag == jie +1:
                break
        if flag != jie+1:
            break
        re = jie+1
    print(re)
def duijiaoxian498():
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    m = len(mat)
    n = len(mat[0])
    print(m,n)
    # for l in range(0,n):
    #     for h in range(0,m):
    #         pass
    # i,j=0,0
    # stk =[]
    # flag = 1
    # while True:
    #     stk.append(mat[i][j])
    #     if flag ==0:
    #         i -=1
    #         j -=1
    #         if i > m-1 or i < 0 or j > n-1 or j < 0:
    #             i +=2
    #             j +=1
    #     if flag ==1:
    #         i += 1
    #         j += 1
    #         if i > m-1 or i < 0 or j > n-1 or j < 0:
    #             i -=1
def num790():
    n = 30
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 5
    else:
        stk = [1,2,5]
        while len(stk)<n:
            l = len(stk)
            stk.append( (2*stk[-1]) + stk[l-3])
        print(stk[-1])
def sort791():
    order = "cba"
    s = "abcd"
    t = len(order)
    stk = [[] for _ in range(t+1)]
    for i in s:
        if i in order:
            stk[order.index(i)].append(i)
        else:
            stk[-1].append(i)
    s = sum(stk,[])
    print(''.join(s))
def customSortString():
    order = "cba"
    s = "abcd"
    dic = defaultdict(int)
    for i,ch in enumerate(order):
        dic[ch] = i+1
    print(dic)
    a = sorted(s,key=(lambda j:dic[j]))
    print(a)
def maximumUnits1710():
    boxTypes = [[5, 10], [2, 5], [4, 7], [3, 9]]
    truckSize = 1
    by = sorted(boxTypes,key=(lambda x:x[1]),reverse=True)
    t = 0
    sum = 0
    for i,j in by:
        if t+i < truckSize:
            t = t + i
            sum = sum + i*j
        else:
            sum = (truckSize-t)*j + sum
            return sum
    return sum
def quanjvdaozhi775():
    nums = [1, 0, 2]
    quan = 0
    jv = 0
    l = len(nums)
    for i in range(l):
        for j in range(i+1,l):
            if j == i+1:
                if nums[i] > nums[j]:
                    jv += 1
            if nums[i] > nums[j]:
                quan += 1
    print(quan,jv)
def pipeizfc792():
    def zixulie(s1,s2):
        l1 = len(s1)
        l2 = len(s2)
        j = 0
        for i in range(l1):
            if s1[i] == s2[j]:
                j+=1
            if j == l2:
                return True
        return False
    s = "dsahjpjauf"
    words = ["ahjpjau", "ja", "ahbwzgqnuk", "tnmlanowax",'dh']
    a = [zixulie(s,word) for word in words]
def zixuliekuan891():
    nums = [3,7,2,3]
    sum = 0
    L = len(nums)
    for i in range(L-1):
        n = 1
        for j in range(i+1,L):
            sum = abs(i-j)*n + sum
            n += 1
    print(sum)
def xiangbin799():
    poured = 5
    query_row = 2
    query_glass = 2

    top = [poured]
    for i in range(1,query_row+1):
        next = [0] * (i+1)
        for j ,vlaue in enumerate(top):
            if vlaue>1:
                next[j] += (vlaue-1)/2
                next[j+1] += (vlaue-1)/2
        top = next
    return min(1,top[query_glass])
def fentang808():
    n = 150
    b = 0
    ab = 0
def shenqishu878():
    n = 12
    a = 2
    b = 8
    stk= []
    i=1
    j=1
    while len(stk) < n:
        if i*a <= j*b:
            if i*a not in stk:
                stk.append(i*a)
            i+=1
        else:
            if j*b not in stk:
                stk.append(j*b)
            j+=1
    print(stk)
def xiaoqiu1742():
    def ss(n):
        s=0
        while n>0:
            s = s + n%10
            n = n //10
        return s
    lowLimit = 1
    highLimit = 10
    dic = defaultdict(int)
    for i in range(lowLimit,highLimit+1):
        dic[ss(i)] += 1
    a = max(dic.values())
    b = dic.get(a)
    print(a,b,dic)
def qujian795():
    nums = [2,9,2,5,6]
    left = 2
    right = 8
def wenzi809():
    S = "heeellooo"
    words = ["hello", "hi", "helo"]
    def pd(s,t):
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                return False
            ch = s[i]
            cns = 0
            while i<len(s) and s[i]==ch:
                cns +=1
                i +=1
            cnt = 0
            while j<len(t) and t[j]==ch:
                cnt += 1
                j += 1
            if cns < cnt:
                return False
            if cnt != cns and cns < 3:
                return False
        return i == len(s) and j == len(t)
    a = sum([pd(S,word) for word in words])
    print(a)
def xifentu882():
    pass
def shuzu813():
    nums = [9, 1, 2, 3, 9]
    k = 3
    l = len(nums)
    print(nums[0:l-k+1])
    print(nums[l-k+1:l-k+2],nums[l-k+2:l])
    t = l-k+1
def zifu1758():
    s="10010100"
    s = list(s)
    i = 1
    f = 0
    ch = s[0]
    while i < len(s):
        if ch == s[i]:
            s[i] = '1' if s[i] == '0' else '0'
            f+=1
        ch=s[i]
        i+=1
    print(s,f,len(s)-f)
def zhan895():
    freq = defaultdict(int)
    group = defaultdict(list)
    freq[5]+=1
    group[freq[5]].append(5)
    freq[7] += 1
    group[freq[7]].append(7)
    print(freq,group)
def zifu1769():
    boxes = "001011"
    L = len(boxes)
    answer = [0 for _ in range(L)]
    for i in range(L):
        for j in range(L):
            if i != j and boxes[j]=='1':
                answer[i] = answer[i] + (abs(i-j))
    print(answer)
def zifu1769_2():
    boxes = "001011"
    left, right, operations = int(boxes[0]), 0, 0
    for i in range(1, len(boxes)):
        if boxes[i] == '1':
            right += 1
            operations += i
    res = [operations]
    for i in range(1, len(boxes)):
        operations += left - right
        if boxes[i] == '1':
            left += 1
            right -= 1
        res.append(operations)
    return res
def erfen705():
    nums = [-1, 0, 3, 5, 9, 12]
    target = 9
    l, r = 0, len(nums) - 1
    while l<r:
        m = (l + r) // 2
        if nums[m] > target:
            r = m
        if nums[m] < target:
            l = m+1
        print(l,r,m)
        if nums[m] == target:
            return m
    return -1
def xiangqi1812():
    coordinates = 'a1'
    return (ord(coordinates[0])+int(coordinates[1]))%2 != 0
def kuai1691():
    cuboids = [[50, 45, 20], [95, 37, 53], [45, 23, 12],[18,48,49]]
    for c in cuboids:
        c.sort()
    cuboids.sort(reverse=True)
    l = len(cuboids)
    h = [0] * l
    print(cuboids)
    for i in range(l):
        for j in range(i):
            print(i,j)
            if cuboids[j][1] >= cuboids[i][1] and cuboids[j][2] >= cuboids[i][2]:
                h[i] = max(h[i], h[j])
        h[i] += cuboids[i][2]
        print(h)
    print(max(h))
def min1827(nums):
    s = 0  # 操作数
    t = 0  # 临时变量
    l = len(nums)
    for i in range(l):
        if nums[i] <= t:
            t = t + 1
            s = s + t - nums[i]
        else:
            t = nums[i]
    return s
from collections import Counter
def beatiful1781():
    s = "wxrplttlrovaylgubhahanxjulhocnxehqdypxrpyubbwjxkusgrkyiefacbefygirmgcsciuevxbrtxqyubvbmrevlddgsedehaiqhduugiomavjcukqyzbinvrfquaoycqddhagwebyrqpsbyctthauimmowwsfyavppdgsllozkywhteyqhzgtwdznvgexjzlemfqnyjywczlxstngfvithbutbqmozpcizrqbhlangmbzubfqolccjivjjfvlhqpsksrumgvjmexlzcwiadjdbviazhpnpubsjljidqlcyctydhftrkojwfvifqukhiavsrczcteybmariedzpjttqvzgzktfmeoyokakryfyeladetdepzqzbmtsbdqeddqlmiqfjhbfltqdtxq"
    re = 0
    le = len(s)
    for i in range(0,le):
        count = defaultdict(int)
        for j in range(i,le):
            count[s[j]] += 1
            re = re + max(count.values()) - min(count.values())
    print(re)
def zhongweishu():
    nums1 = [100001]
    nums2 = [100000]
    i = 0
    j = 0
    l = []
    l1 = len(nums1)
    l2 = len(nums2)
    l3 = (l1 + l2) // 2 + 1
    if not nums1:
        if l2%2 ==0:
             return (nums2[l2//2]+nums2[l2//2-1])/2
        else:
            return nums2[l2//2]
    if not nums2:
        if l1%2 ==0:
             return (nums1[l1//2]+nums1[l1//2-1])/2
        else:
            return nums1[l1//2]

    while len(l) < l3:
        if i<l1 and nums1[i] <= nums2[j]:
            l.append(nums1[i])
            i += 1
        else:
            l.append(nums2[j])
            j += 1
    if (l1 + l2) % 2 == 0:
        return (l[-1] + l[-2]) / 2
    else:
        return l[-1]
def shuzu1785():
    import math
    nums = [1,-1,1]
    limit = 3
    goal = -4
    i = 0
    s = sum(nums)
    t = goal - s
    i = math.ceil(abs(t) / limit)
    print(i)
def shuzu1764():   # while--else--方法，很值得学习
    groups = [[10,-2],[1,2,3,4]]
    nums = [1,2,3,4,10,-2]
    k = 0
    for i in groups:
        while k < len(nums):
            if nums[k:k+len(i)] == i:
                k += len(i)
                break
            k += 1
        else:
            return False
    return True
def jiaohuan1703():
    nums = [1,0,0,0,0,0,1,1]
    k = 3
    chedibuhui = 'g'
def shizi1753():
    a = 1
    b = 8
    c = 8
    s = []
    s.append(a)
    s.append(b)
    s.append(c)
    s.sort()
    if s[2] == s[1]:
        return s[2]
    k = 0
    while s[0] + s[1] > s[2]:
        s[0] -= 1
        s[1] -= 1
        k += 1
    if s[0] + s[1] <= s[2]:
        return k+s[0]+s[1]
def caozuo2011():
    operations = ["--X", "X++", "X++"]
    k=0
    return sum(1 if s == "X++" or s == "++X" else -1 for s in operations)
def chuan1754():
    word1 = "cabaa"
    word2 = "bcaaa"
    merge = []
    i,j = 0,0
    l1 = len(word1)
    l2 = len(word2)
    while i < l1 or j < l2:
        if i < l1 and word1[i:] > word2[j:]:
            merge.append(word1[i])
            i += 1
        else:
            merge.append(word2[j])
            j += 1
    print(merge)
def tonggou1759():
    s = "zzzzz"
    i,j = 0,0
    suu = 0
    t = 1
    l = len(s)
    while i < l and j < l:
        if s[i] == s[j]:
            suu = suu + t
            j = j + 1
            t = t + 1
        else:
            t = 1
            i = j
    print(suu)
def OX2027():
    s = "OXOXXOOX"
    res = 0
    i = 0
    while i < len(s):
        if s[i] == 'X':
            i = i + 3
            res += 1
        else:
            i = i + 1
    print(res)
def qianhouzhui1750():
    s = "cabaabac"
    l = len(s)-1
    i = 0
    while s[i] == s[l] and i < l:
        while s[i] == s[i+1]:
            i += 1
        while s[l] == s[l-1]:
            l -= 1
        i += 1
        l -= 1
    if l-i+1 <0:
        return 0
    else:
        return l-i+1
def shuzu2023():
    nums1 = [3,1]
    nums2 = [2,3]
    nums3 = [1,2]

    nums1 = set(nums1)
    nums2 = set(nums2)
    nums3 = set(nums3)
    s = []
    dic = defaultdict(int)
    for i in nums1:
        dic[i] += 1
    for i in nums2:
        dic[i] += 1
    for i in nums3:
        dic[i] += 1
    for k,v in dic.items():
        if v >= 2:
            s.append(k)
    return(s)
import bisect
class ExamRoom(object):
    def __init__(self, N):
        self.N = N
        self.students = []

    def seat(self):
        # Let's determine student, the position of the next
        # student to sit down.
        if not self.students:
            student = 0
        else:
            # Tenatively, dist is the distance to the closest student,
            # which is achieved by sitting in the position 'student'.
            # We start by considering the left-most seat.
            dist, student = self.students[0], 0
            for i, s in enumerate(self.students):
                if i:
                    prev = self.students[i-1]
                    # For each pair of adjacent students in positions (prev, s),
                    # d is the distance to the closest student;
                    # achieved at position prev + d.
                    d = (s - prev) // 2
                    if d > dist:
                        dist, student = d, prev + d

            # Considering the right-most seat.
            d = self.N - 1 - self.students[-1]
            if d > dist:
                student = self.N - 1

        # Add the student to our sorted list of positions.
        bisect.insort(self.students, student)
        return student

    def leave(self, p):
        self.students.remove(p)
def zuowei2037():
    seats = [4, 1, 5, 9]
    students = [1, 3, 2, 6]
    seats.sort()
    students.sort()
    s = 0
    for i in range(0,len(seats)):
        s = s + abs(seats[i]-students[i])
    print(s)
def zimu2351():
    s = "abccbaacz"
    # 数组方法
    # t = []
    # for i in s:
    #     if i in t:
    #         return i
    #     t.append(i)
    # 字典方法
    dic = defaultdict(int)
    for i in s:
        dic[i] += 1
        if dic[i] == 2:
            return i
from heapq import heappush,heappop
def dingdan1081():
    # orders = [[10, 5, 0], [15, 2, 1], [25, 1, 1], [30, 4, 0]]
    orders = [[10, 5, 0], [5, 5, 0],[15, 5, 0], [10, 5, 1], [5, 5, 1]]
    buy, sell = [], []
    for p, a, t in orders:
        if t == 0:
            while a and sell and sell[0][0] <= p:
                x, y = heappop(sell)
                if a >= y:
                    a -= y
                else:
                    heappush(sell, (x, y - a))
                    a = 0
            if a:
                heappush(buy, (-p, a))
        else:
            while a and buy and -buy[0][0] >= p:
                x, y = heappop(buy)
                if a >= y:
                    a -= y
                else:
                    heappush(buy, (x, y - a))
                    a = 0
            if a:
                heappush(sell, (p, a))
    mod = 10 ** 9 + 7
    print(sum(v[1] for v in buy + sell) % mod)
def zifushuzi2042():
    s = "sunset is at 7 51 pm overnight lows will be in the low 50 and 60 s"
    s = s.split(' ')
    t = -1
    for i in s:
        if i.isdigit():
            if int(i) > t:
                t = int(i)
            else:
                return False
    return True
def areNumbersAscending(s: str) -> bool:
    lst = -float('inf')
    sp = s.split(' ')
    for c in sp:
        if c.isdigit():
            if int(c) <= lst:
                return False
            lst = int(c)
    return True
def zuida1802():
    n = 6
    index = 1
    maxSum = 10
    pass
def countPairs1803():
    nums = [1, 4, 2, 7]
    low = 2
    high = 16
    f = 0
    l = len(nums)
    for i in range(l):
        for j in range(i+1,l):
            if low<= nums[i] ^ nums[j] <= high:
                f += 1
    print(f)
def geweihe2108():
    num = 15
    f = 0
    for i in range(1, num + 1):
        x = i
        s = 0
        while x:
            s = s + x % 10
            x = x // 10
        if s % 2 == 0:
            f += 1
    return f
def minOperations():
    nums = [5,5,8,1,1]
    x = 10
    n = len(nums)
    total = sum(nums)

    if total < x:
        return -1

    right = 0
    lsum, rsum = 0, total
    ans = n + 1
    for left in range(-1, n - 1):
        if left != -1:
            lsum += nums[left]
        while right < n and lsum + rsum > x:
            rsum -= nums[right]
            right += 1
        if lsum + rsum == x:
            ans = min(ans, (left + 1) + (n - right))

    return -1 if ans > n else ans
def minop1806():
    n = 4
    f = 0
    perm = list(range(n))
    ans = perm.copy()
    arr = list(range(1,n+1))
    print(perm,arr,ans)
    while arr != ans:
        for i in range(n):
            if i%2 == 0:
                arr[i] = perm[int(i/2)]
            else:
                arr[i] = perm[int(n/2 + (i-1)/2)]
        perm = arr.copy()
        f += 1
    return f
if __name__ == '__main__':
    print(minop1806())