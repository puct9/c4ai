#include <iostream>
#include <sstream>
#include <cctype>
#include "C4Game.h"


bool C4Game::READY = false;
unsigned long long C4Game::BITSH[42] = { 0 };


void C4Game::init_variables()
{
    if (READY) return;
    for (int i = 0; i < 42; i++)
    {
        BITSH[i] = (unsigned long long)1 << i;
    }
    READY = true;
}

void C4Game::repr_ull_as_7x6(unsigned long long v)
{
    for (int row = 5; row >= 0; row--)
    {
        for (int col = 0; col < 7; col++)
        {
            if (BITSH[row * 7 + col] & v)
                std::cout << " 1";
            else
                std::cout << " 0";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

C4Game::C4Game() : move_history{}, pcs_o(0), pcs_x(0), move_n(0), start_n(0)
{
}

C4Game::C4Game(std::string & posstr) : move_history{}, pcs_o(0), pcs_x(0), move_n(0)
{
    /*
    BOARD INDEXES FOR BITSHIFT
    0   1   2   3   4   5   6
    ____________________________
    35  36  37  38  39  40  41  |   5
    28  29  30  31  32  33  34  |   4
    21  22  23  23  25  26  27  |   3
    14  15  16  17  18  19  20  |   2
    7   8   9   10  11  12  13  |   1
    0   1   2   3   4   5   6   |   0
    */

    int grid_n = 0;
    bool error = false;
    for (auto& c : posstr)
    {
        if (grid_n >= 42)
        {
            // error
            error = true;
            break;
        }
        char ch = tolower(c);
        if (isdigit(ch))
        {
            grid_n += ch - '0';
            continue;
        }
        else if (ch == 'x' || ch == 'o')
        {
            int col = grid_n % 7;
            int row = 5 - grid_n / 7;
            if (ch == 'x')
                pcs_x |= BITSH[row * 7 + col];
            else
                pcs_o |= BITSH[row * 7 + col];
            grid_n++;
            move_n++;
        }
        else  // slash (/)
        {
            if (grid_n % 7 != 0)
            {
                // error
                error = true;
                break;
            }
        }
    }
    if (error)
    {
        // reset everything
        pcs_o = 0;
        pcs_x = 0;
        move_n = 0;
    }
    start_n = move_n;
}

bool * C4Game::LegalMoves()
{
    auto occupied = pcs_x | pcs_o;
    // check top row
    for (int i = 0; i < 7; i++)
    {
        legal[i] = (occupied & BITSH[35 + i]) ? false : true;
    }

    return legal;
}

bool C4Game::CheckWin()
{
    /*
    BOARD INDEXES FOR BITSHIFT
    0   1   2   3   4   5   6
    ____________________________
    35  36  37  38  39  40  41  |   5
    28  29  30  31  32  33  34  |   4
    21  22  23  23  25  26  27  |   3
    14  15  16  17  18  19  20  |   2
    7   8   9   10  11  12  13  |   1
    0   1   2   3   4   5   6   |   0
    */
    // only checks for a win from the last player
    if (move_n < 7) return false;
    auto pcs = move_n % 2 == 1 ? pcs_x : pcs_o;
    auto last_placed = move_history[move_n - 1];

    // look up and down
    int vertical = 0;
    // looking up
    for (int i = 1; last_placed + i * 7 < 42; i++)
    {
        if (!(pcs & BITSH[last_placed + i * 7]))
            break;
        vertical++;
    }
    // looking down
    for (int i = 1; last_placed - i * 7 >= 0; i++)
    {
        if (!(pcs & BITSH[last_placed - i * 7]))
            break;
        vertical++;
    }
    if (vertical >= 3)
        return true;

    // look left and right
    int horizontal = 0;
    int column = last_placed % 7;
    // looking left
    for (int i = 1; i <= column; i++)
    {
        if (!(pcs & BITSH[last_placed - i]))
            break;
        horizontal++;
    }
    // looking right
    for (int i = 1; i <= 6 - column; i++)
    {
        if (!(pcs & BITSH[last_placed + i]))
            break;
        horizontal++;
    }
    if (horizontal >= 3)
        return true;

    // look at the (/) diagonal
    int diagonal = 0;
    int row = last_placed / 7;
    // look up and right
    for (int i = 1; i <= (5 - row < 6 - column ? 5 - row : 6 - column); i++)
    {
        if (!(pcs & BITSH[last_placed + i * 8]))
            break;
        diagonal++;
    }
    // look down and left
    for (int i = 1; i <= (row < column ? row : column); i++)
    {
        if (!(pcs & BITSH[last_placed - i * 8]))
            break;
        diagonal++;
    }
    if (diagonal >= 3)
        return true;

    // look at the (\) diagonal
    diagonal = 0;
    // look up and left
    for (int i = 1; i <= (5 - row < column ? 5 - row : column); i++)
    {
        if (!(pcs & BITSH[last_placed + i * 6]))
            break;
        diagonal++;
    }
    // look down and right
    for (int i = 1; i <= (row < 6 - column ? row : 6 - column); i++)
    {
        if (!(pcs & BITSH[last_placed - i * 6]))
            break;
        diagonal++;
    }
    if (diagonal >= 3)
        return true;

    return false;
}

int C4Game::GameOver()
{
    // -1 ongoing
    // 0 drawn
    // 1 won
    if (CheckWin()) return 1;
    if (move_n == 42) return 0;
    return -1;
}

int C4Game::GetIndFromCol(int col)
{
    auto occupied = pcs_x | pcs_o;
    for (int row = 0; row < 6; row++)
    {
        if (!(occupied & BITSH[row * 7 + col]))
            return row * 7 + col;
    }
    return -1;  // this should never be reached, and if it is, then it is an illegal move
}

void C4Game::PlayMove(int col)
{
    auto mv = GetIndFromCol(col);
    if (!(move_n % 2))
        pcs_x |= BITSH[mv];
    else
        pcs_o |= BITSH[mv];
    // housekeeping
    move_history[move_n++] = mv;  // can't do this one liner in python eh?
}

void C4Game::Show()
{
    for (int r = 5; r >= 0; r--)
    {
        for (int c = 0; c < 7; c++)
        {
            if (BITSH[r * 7 + c] & pcs_x)
                std::cout << "| X ";
            else if (BITSH[r * 7 + c] & pcs_o)
                std::cout << "| O ";
            else
                std::cout << "|   ";
        }
        std::cout << "|" << std::endl;
    }
    std::cout << "-----------------------------\n  0   1   2   3   4   5   6" << std::endl;
}

void C4Game::UndoMove()
{
    if (!(move_n - start_n))
        return;
    if (!(move_n % 2)) // it is currently X to move, so the last move was by O
        pcs_o ^= BITSH[move_history[--move_n]];
    else
        pcs_x ^= BITSH[move_history[--move_n]];
    move_history[move_n] = 0; // unnecessary, just for the sake of my sanity lmao
}

float * C4Game::GetPositionArray()
{
    this->WritePositionToArray(this->position_array);
    return this->position_array;
}

void C4Game::WritePositionToArray(float * arr)
{
    // arr size should be 7 * 6 * 3 = 126
    /*
    BOARD INDEXES FOR BITSHIFT [A]
    35  36  37  38  39  40  41
    28  29  30  31  32  33  34
    21  22  23  23  25  26  27
    14  15  16  17  18  19  20
    7   8   9   10  11  12  13
    0   1   2   3   4   5   6

    REFERENCE [B]
    0   7   14  21  28  35
    1   8
    2   9
    3   10
    4   11
    5   12
    6   ETC

    USE INDEXES * 3 [C]
    0   1   2   3   4   5
    6   7   8   9   10  11
    12  13  14
    18
    24
    30
    36

    Conversion from [A] -> [C] using R, C
    R' = C
    C' = R
    */
    for (int i = 0; i < 42; i++)
    {
        int new_r = i % 7;
        int new_c = i / 7;
        int new_ind = (new_r * 6 + new_c) * 3;
        // CHANNEL ORDER: TURN | X | O
        // TURN
        arr[new_ind] = move_n % 2 == 0 ? 1.0f : 0.0f;
        // X
        arr[new_ind + 1] = BITSH[i] & pcs_x ? 1.0f : 0.0f;
        // O
        arr[new_ind + 2] = BITSH[i] & pcs_o ? 1.0f : 0.0f;
    }
}


C4Game::~C4Game()
{
}
