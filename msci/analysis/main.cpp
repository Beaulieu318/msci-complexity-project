//
//  main.cpp
//  louvain
//
//  Created by Sebastian Lee on 06/02/2018.
//  Copyright © 2018 Sebastian Lee. All rights reserved.
//

#include <iostream>
#include <map>
#include <vector>
#include <string>

using namespace std;

void common_stores(const vector<vector<int>> & stores_visited, vector<vector<int>> & mutual_stores, map<vector<int>, vector<int>> & mutual_dictionary){
    for(int i=0; i<stores_visited.size(); i++){
        for(int j=i; j<stores_visited.size(); j++){
            vector<int> common;
            for(int store_a=0; store_a<stores_visited[i].size(); store_a++){
                for(int store_b=0; store_b<stores_visited[j].size(); store_b++){
                    if(stores_visited[i][store_a] == stores_visited[j][store_b]){
                        common.push_back(stores_visited[i][store_a]);
                    }
                }
            }
            mutual_stores[i][j]=common.size();
            vector<int> key = {i,j};
            mutual_dictionary[key]=common;
        }
    }
}




int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    vector<vector<int>> stores_visited;
    vector<vector<int>> mutual_stores;
    map<vector<int>, vector<int>> mutual_dictionary;
    common_stores(stores_visited, mutual_stores, mutual_dictionary);
    for(int i=0; i<3; i++){
        for(int j=0; j<mutual_stores[i].size(); j++){
            cout<<mutual_stores[i][j]<<" ";
        }
        cout<< " \n";
    }
    return 0;
}
