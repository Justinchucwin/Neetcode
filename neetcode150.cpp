#include <iostream>
#include <algorithm>
#include <vector> 
#include <string>
#include <cstdlib>
#include <queue>
using namespace std;

bool hasDuplicate(vector<int>& nums){
    sort(nums.begin(),nums.end());
    for (int i=1;i<nums.size();i++){
        if (nums[i]==nums[i-1]){
            return true;
        }
    }
    return false;
}

string encode(vector<string> strs){
    string encoded;
    for (const string& s:strs){
        encoded+=to_string(s.size())+"#"+s;
    }
    return encoded;
}

vector<string> decode(const string& s){
    vector<string> res;
    int i=0;
    int j=0;
    while (j<s.size()){
        if (s[j]=='#'){
            int length=stoi(s.substr(i,j-i));
            i=j+1;
            res.push_back(s.substr(i,length));
            i+=length;
            j+=length;
        }
        j++;
    }
    return res;
}

int lastStoneWeight (vector<int>& stones){
    // while (stones.size()>1){ sorting method time complexity o(n^2logn)
    //     sort(stones.begin(),stones.end());
    //     int difference=abs(stones[stones.size()-1]-stones[stones.size()-2]);
    //     for (int i=0;i<2;i++){
    //         stones.pop_back();
    //     }
    //     if (difference!=0){
    //         stones.push_back(difference);
    //     }
    // }
    // if (stones.size()==0){
    //     return 0;
    // }
    // return stones[0];

    priority_queue<int> maxheap;
    for (int w:stones){
        maxheap.push(w);
    }
    while (maxheap.size()>1){
        int first=maxheap.top();
        maxheap.pop();
        int second=maxheap.top();
        maxheap.pop();
        if (first>second){
            maxheap.push(first-second);
        }
    }
    maxheap.push(0);
    return maxheap.top();
}





int main(){
    // vector<int> nums={1,2,3,3};
    // if (hasDuplicate(nums)){
    //     cout<<"True"<<endl;
    // }
    // else{
    //     cout<<"False"<<endl;
    // }
    vector<int> weights={2,3,6,2,4};
    cout<<lastStoneWeight(weights)<<endl;
    // string encoded=encode(strs);
    // strs=decode(encoded);
    // for (int i=0;i<strs.size();i++){
    //     if (i==strs.size()-1){
    //         cout<<strs[i]<<endl;
    //     }
    //     else{
    //         cout<<strs[i]<<",";
    //     }
    // }
    // return 0;
}