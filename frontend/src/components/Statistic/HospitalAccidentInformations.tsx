import { useState } from 'react';
import { hospitalAccidentDataGridColumns } from '~/data/statistic';
import { useAppSelector } from '~/lib/hooks/redux';
import { useHospitalAccidentInformationsQuery } from '~/lib/hooks/useApi';
import { HospitalAccidentInformationDto } from '~/types/common.dto';
import AccidentInformationEditModal from '../Modal/AccidentInformationEditModal';
import AccidentInformations from './AccidentInformations';

const HospitalAccidentInformations = () => {
  const hospitalId = useAppSelector((state) => state.member).value
    .hospitalId?.[0];
  const [pageNumber, setPageNumber] = useState<number>(0);
  const [pageSize, setPageSize] = useState<number>(5);
  const [selectedAccident, setSelectedAccident] =
    useState<HospitalAccidentInformationDto | null>(null);
  const { data: accidentInformations, isLoading } =
    useHospitalAccidentInformationsQuery(pageNumber, pageSize, hospitalId);

  return (
    <>
      <AccidentInformations
        accidentInformations={accidentInformations}
        pageNumber={pageNumber}
        setPageNumber={setPageNumber}
        pageSize={pageSize}
        setPageSize={setPageSize}
        columns={hospitalAccidentDataGridColumns}
        onRowClick={({ row }: { row: HospitalAccidentInformationDto }) =>
          setSelectedAccident(row)
        }
        loading={isLoading}
      />
      <AccidentInformationEditModal
        selectedAccident={selectedAccident}
        onClose={() => setSelectedAccident(null)}
      />
    </>
  );
};

export default HospitalAccidentInformations;
