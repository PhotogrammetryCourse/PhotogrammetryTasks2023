#include "rasserts.h"

int debugPoint(int line)
{
    if (line < 0)
        return 0;

    // You can put breakpoint at the following line to catch any rassert failure:
    return line;
}
