#ifndef LM_H
#define LM_H

#include <iostream>  

using namespace std;
//template <typename T>





//int length(T& matrix)
//{
//    //cout << sizeof(matrix[0]) << endl;
//    //cout << sizeof(matrix) << endl;
//    return end(matrix) - begin(matrix);
//}





class LM{

public:
  LM() { cout << "LM()" << endl; }
  LM(const LM& src);
  ~LM() { cout << "~LM()" << endl; }
  
};



#endif
