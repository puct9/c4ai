#ifndef C4UCT_C4_GAME_H
#define C4UCT_C4_GAME_H

#include <string>

/*
BOARD INDEXES FOR BITSHIFT
35  36  37  38  39  40  41
28  29  30  31  32  33  34
21  22  23  23  25  26  27
14  15  16  17  18  19  20
7   8   9   10  11  12  13
0   1   2   3   4   5   6
*/


class C4Game
{
private:
    int move_n;
    int start_n; // the position might be manually set
    int move_history[42];
    unsigned long long pcs_x;
    unsigned long long pcs_o;
    
    // one must avoid hanging pointers
    float position_array[126];
    bool legal[7];

public:
    static bool READY;
    static unsigned long long BITSH[42];
    static void init_variables();
    static void repr_ull_as_7x6(unsigned long long v);

    C4Game();
    C4Game(std::string& posstr);

    bool* LegalMoves();
    bool CheckWin();
    int GameOver();
    int GetIndFromCol(int col);
    void PlayMove(int col);
    void Show();
    void UndoMove();
    float* GetPositionArray();
    void WritePositionToArray(float* arr);

    int GetMoveNum() { return this->move_n; };

    ~C4Game();
};

#endif
