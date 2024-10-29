import { useState } from 'react';
import { globalAccidentDataGridColumns } from '~/data/statistic';
import { useGlobalAccidentInformationsQuery } from '~/lib/hooks/useApi';
import AccidentInformations from './AccidentInformations';

const GlobalAccidentInformations = () => {
  const [pageNumber, setPageNumber] = useState<number>(0);
  const [pageSize, setPageSize] = useState<number>(5);
  const { data: accidentInformations, isLoading } =
    useGlobalAccidentInformationsQuery(pageNumber, pageSize);

  return (
    <AccidentInformations
      accidentInformations={accidentInformations}
      pageNumber={pageNumber}
      setPageNumber={setPageNumber}
      pageSize={pageSize}
      setPageSize={setPageSize}
      columns={globalAccidentDataGridColumns}
      loading={isLoading}
    />
  );
};

export default GlobalAccidentInformations;
