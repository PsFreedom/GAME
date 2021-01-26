typedef struct{
    char *start;
    char *end;
} async_list_t;

async_list_t *async_list;
int tot_entry;
int cur_entry;

void async_list_init(int total)
{
    printf("\tgpuGAME >> %s - %d\n", __FUNCTION__, total);
    async_list = (async_list_t*)malloc(sizeof(async_list_t)*total);
    tot_entry = total;
    cur_entry = 0;
}

bool async_list_check(char *start, char *end)
{
    for( int i=0; i<cur_entry; i++ ){
        if( async_list[i].start <= end && 
            async_list[i].end   > start){
            return true;
        }
    }
    return false;
}

void async_list_add(char *start, uint32_t count)
{
    if( cur_entry >= tot_entry || async_list_check(start, start + count) ){
    //    printf("\tgpuGAME >> %s - DeviceSynchronize\n", __FUNCTION__);
        assert( cudaDeviceSynchronize() == cudaSuccess);
        cur_entry = 0;
    }
    async_list[cur_entry].start = start;
    async_list[cur_entry].end   = start + count;
//    printf("\tgpuGAME >> %s - %p %p\n", __FUNCTION__, async_list[cur_entry].start, async_list[cur_entry].end);
    cur_entry++;
}

