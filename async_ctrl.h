typedef struct{
    cudaStream_t cpySteam;
    char *start;
    char *end;
} async_list_t;

async_list_t *async_list;
int tot_entry;
int sync_cntr;

void async_list_init(int total)
{
    sync_cntr = 0;
    tot_entry = total;
    async_list = (async_list_t*)malloc(sizeof(async_list_t)*total);
    for(int i=0; i<total; i++){
        cudaStreamCreate( &async_list[i].cpySteam );
        async_list[i].start = 0;
        async_list[i].end   = 0;
    }
}

int async_list_check(char *start, char *end)
{
    int freeStream = tot_entry;
    for( int i=0; i<tot_entry; i++ ){
        if( async_list[i].start <= end && 
            async_list[i].end   > start){
            return i;
        }
        if( freeStream == tot_entry && cudaStreamQuery(async_list[i].cpySteam) == cudaSuccess ){
            freeStream = i;
        }
    }
    return freeStream;
}

int async_list_add(char *start, uint32_t count)
{
    int stream_no = async_list_check(start, start + count);
    if( stream_no >= tot_entry ){
        assert( cudaDeviceSynchronize() == cudaSuccess);
        stream_no = 0;
    }
    async_list[stream_no].start = start;
    async_list[stream_no].end   = start + count;
    return stream_no;
}

