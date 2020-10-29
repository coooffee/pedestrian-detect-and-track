//
// Created by didi on 2020/8/31.
//
#include <iostream>
#include <vector>
using namespace std;

int minDistance(string word1, string word2) {
//    vector<vector<int>> dp_table(word1.length()+1, vector<int>(word2.length()+1,0));
    int dp_table[word1.length()+1][word2.length()+1];
    for(int i=0;i<word1.length()+1;i++)
        dp_table[i][0] = i;
    for(int i=0;i<word2.length()+1;i++)
        dp_table[0][i] = i;
    for(int i=1;i<word1.length()+1;i++){
        for(int j=1;j<word2.length()+1;j++){
            if(word1[i-1]==word2[j-1])
                dp_table[i][j] = dp_table[i-1][j-1];
            else{
                dp_table[i][j] = min(dp_table[i][j-1]+1, dp_table[i-1][j-1]+1);
                dp_table[i][j] = min(dp_table[i][j], dp_table[i-1][j]+1);
            }
        }
    }
    return dp_table[word1.length()][word2.length()];
}

#include <unordered_map>
#include <list>

class LRUCache{
    unordered_map<int, list<pair<int,int>>::iterator> map;
    list<pair<int,int>> cache;
    int cap;
public:
    explicit LRUCache(int capacity): cap(capacity){}
    int get(int key){
        if (map.find(key)==map.end()){
            return -1;
        }else{
            int res = map[key]->second;
            put(key, res);
            return res;
        }
    }
    void put(int key, int value){
        pair<int,int> kv(key, value);

        if (map.find(key)==map.end()){
            if (cache.size()==cap){
                pair<int, int> last = cache.back();
                cache.pop_back();
                map.erase(last.first);
            }
            cache.emplace_front(kv);
            map[key] = cache.begin();
        }else{
            cache.erase(map[key]);
            cache.emplace_front(kv);
            map[key] = cache.begin();
        }
    }
};

#include <iostream>
#include <omp.h>

int main()
{
    omp_set_num_threads(4);
#pragma omp parallel for default(none)
    for (int i = 0; i < 8; i++)
    {
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
    }
    printf("\n");
}