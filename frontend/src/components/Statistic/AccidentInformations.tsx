/* eslint-disable @typescript-eslint/no-explicit-any */
import { Box, Stack } from '@mui/material';
import {
  DataGrid,
  GridColDef,
  GridPagination,
  gridClasses,
} from '@mui/x-data-grid';
import { Dispatch, SetStateAction } from 'react';

interface AccidentInformationsProps {
  accidentInformations: any;
  pageNumber: number;
  setPageNumber: Dispatch<SetStateAction<number>>;
  pageSize: number;
  setPageSize: Dispatch<SetStateAction<number>>;
  columns: GridColDef<any>[];
  onRowClick?: ({ row }: { row: any }) => void;
  loading: boolean;
}

const AccidentInformations = ({
  accidentInformations,
  pageNumber,
  setPageNumber,
  pageSize,
  setPageSize,
  columns,
  onRowClick,
  loading,
}: AccidentInformationsProps) => {
  return (
    <div className="h-[371px]">
      <DataGrid
        loading={loading}
        rows={accidentInformations?.content}
        columns={columns}
        slots={{
          footer: () => (
            <Stack
              direction="row"
              borderTop={1}
              borderColor="#E0E0E0"
              padding={1}
              paddingY={0}
              gap={2}
            >
              <Box flex="auto" />
              <GridPagination
                rowsPerPage={pageSize}
                page={pageNumber}
                count={accidentInformations?.totalElements || 0}
                onRowsPerPageChange={(e) =>
                  setPageSize(parseInt(e.target.value))
                }
                onPageChange={(_, page) => setPageNumber(page)}
              />
            </Stack>
          ),
        }}
        getRowId={(row) => row.id}
        pageSizeOptions={[5, 10, 50, 100]}
        onRowClick={onRowClick}
        disableRowSelectionOnClick
        sx={{
          '--DataGrid-containerBackground': '#F2F2F2', // 헤더의 배경색 설정
          [`& .${gridClasses.cell}:focus, & .${gridClasses.cell}:focus-within`]:
            {
              outline: 'none',
            },
          [`& .${gridClasses.columnHeader}:focus, & .${gridClasses.columnHeader}:focus-within`]:
            {
              outline: 'none',
            },
        }}
      />
    </div>
  );
};

export default AccidentInformations;
