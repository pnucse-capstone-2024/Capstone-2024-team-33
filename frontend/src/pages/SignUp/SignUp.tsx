import { useExistMemberIdQuery, useMemberCreate } from '@lib/hooks/useApi';
import { Box, Button, Divider, TextField } from '@mui/material';
import { Form, Formik, FormikHelpers, useField } from 'formik';
import { useState } from 'react';
import { FaSearch } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import * as Yup from 'yup';
import HospitalSearchModal from '~/components/Modal/HospitalSearchModal';
import { HospitalType, hospitalTypeLabel } from '~/data/hospital';

export interface Hospital {
  id: number;
  name: string;
  type: HospitalType;
}

export interface SignUpInfo {
  id: string;
  password: string;
  confirmPassword: string;
  hospital: Hospital;
  managerName: string;
  managerPhoneNumber: string;
}

const initialValues: SignUpInfo = {
  id: '',
  password: '',
  confirmPassword: '',
  hospital: { id: 0, name: '', type: 'HOSPITAL' },
  managerName: '',
  managerPhoneNumber: '',
};

const validationSchema = Yup.object({
  id: Yup.string()
    .matches(
      /^[a-zA-Z0-9-_]{8,32}$/,
      '아이디는 8자~32자의 영문 대소문자, 숫자, 특수문자(-,_)로 이루어져야합니다',
    )
    .required('ID를 입력해주세요'),
  password: Yup.string()
    .matches(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])[a-zA-Z0-9!@#$%^&*]{8,32}$/,
      '비밀번호는 8자~32자의 영문 대소문자, 숫자를 모두 포함하고 있어야 합니다',
    )
    .required('비밀번호를 입력해주세요'),
  confirmPassword: Yup.string()
    .oneOf([Yup.ref('password')], '비밀번호가 일치하지 않습니다')
    .required('비밀번호를 다시 입력해주세요'),
  hospital: Yup.object().shape({
    id: Yup.number().min(1, '병원을 검색하여 선택해주세요'),
  }),
  managerName: Yup.string().required('관리자 이름을 입력해주세요'),
  managerPhoneNumber: Yup.string()
    .matches(
      /^010([0-9]{3,4})([0-9]{4})$/,
      '전화번호는 010XXXXXXXX의 형식이어야 합니다.',
    )
    .required('관리자 전화번호를 입력해주세요'),
});

const SignUp = () => {
  const navigate = useNavigate();
  const { mutate: createMember, isPending: isCreateMemberPending } =
    useMemberCreate();
  const { mutate: isExistId, isPending: isCheckExistMemberPending } =
    useExistMemberIdQuery();
  const [idAvailable, setIdAvailable] = useState<boolean>(false);
  const [openHospitalSearchModal, setOpenHospitalSearchModal] =
    useState<boolean>(false);

  const handleSubmit = async (
    values: SignUpInfo,
    { setSubmitting }: FormikHelpers<SignUpInfo>,
  ) => {
    createMember(
      {
        loginId: values.id,
        loginPassword: values.password,
        hospitalId: values.hospital.id,
        adminName: values.managerName,
        adminTel: values.managerPhoneNumber,
      },
      {
        onSuccess: () => {
          alert('회원가입이 완료되었습니다.');
          navigate('/');
        },
        onError: () => {
          alert('회원가입에 실패하였습니다.');
        },
      },
    );
    setSubmitting(false);
  };

  const IdField = () => {
    const [field, meta] = useField('id');

    const checkIdAvailability = async () => {
      isExistId(field.value, {
        onSuccess: () => {
          setIdAvailable(true);
        },
        onError: () => {
          setIdAvailable(false);
        },
      });
    };
    return (
      <div>
        <div className="flex gap-2">
          <TextField
            {...field}
            type="text"
            label="아이디"
            className="flex-1 flex-grow"
            error={meta.touched && Boolean(meta.error)}
            helperText={meta.touched && meta.error}
            FormHelperTextProps={{ sx: { color: 'success.main' } }}
            onChange={(e) => {
              field.onChange(e);
              setIdAvailable(false);
            }}
          />
          <Button
            type="button"
            variant="outlined"
            onClick={checkIdAvailability}
            disabled={
              isCheckExistMemberPending || (meta.touched && Boolean(meta.error))
            }
            className="h-14"
          >
            중복 확인
          </Button>
        </div>
        {idAvailable && (
          <p className="pl-3 pt-2 text-xs text-primary">
            사용 가능한 아이디입니다.
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-full items-center justify-center">
      <Formik
        initialValues={initialValues}
        validationSchema={validationSchema}
        onSubmit={handleSubmit}
      >
        {({
          isSubmitting,
          values,
          errors,
          touched,
          handleChange,
          handleBlur,
          setFieldValue,
        }) => (
          <Form className="flex w-full max-w-sign flex-col gap-2 rounded-lg bg-white px-10 py-16 shadow-md">
            <div className="flex flex-col gap-2">
              <Box className="py-4">
                <p className="mb-1 text-xl">회원가입</p>
                <Divider />
              </Box>
              <div className="flex flex-col gap-8">
                <div className="flex flex-col gap-4">
                  <p className="text-xl">계정 정보</p>
                  <IdField />
                  <div className="flex gap-2">
                    <TextField
                      name="password"
                      type="password"
                      label="비밀번호"
                      className="flex-1 flex-grow"
                      value={values.password}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      error={touched.password && Boolean(errors.password)}
                      helperText={touched.password && errors.password}
                    />
                    <TextField
                      name="confirmPassword"
                      type="password"
                      label="비밀번호 확인"
                      className="flex-1 flex-grow"
                      value={values.confirmPassword}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      error={
                        touched.confirmPassword &&
                        Boolean(errors.confirmPassword)
                      }
                      helperText={
                        touched.confirmPassword && errors.confirmPassword
                      }
                    />
                  </div>
                </div>
                <div className="flex flex-col gap-4">
                  <p className="text-xl">병원 정보</p>
                  <div className="flex gap-2">
                    <TextField
                      type="text"
                      placeholder="병원명"
                      className="flex-grow"
                      value={values.hospital.name}
                      disabled={true}
                      error={
                        touched.hospital?.id && Boolean(errors.hospital?.id)
                      }
                      helperText={touched.hospital?.id && errors.hospital?.id}
                    />
                    <Button
                      type="button"
                      variant="outlined"
                      className="!min-w-[56px] max-w-[56px] !border-2"
                      onClick={() => setOpenHospitalSearchModal(true)}
                    >
                      <FaSearch className="h-9 w-9" />
                    </Button>
                  </div>
                  <TextField
                    type="text"
                    placeholder="병원 유형"
                    className="flex-grow"
                    value={hospitalTypeLabel[values.hospital.type]}
                    disabled={true}
                    error={touched.hospital?.id && Boolean(errors.hospital?.id)}
                    helperText={touched.hospital?.id && errors.hospital?.id}
                  />
                </div>
                <div className="flex flex-col gap-4">
                  <p className="text-xl">관리자 정보</p>
                  <div className="flex gap-2">
                    <TextField
                      type="text"
                      label="관리자 이름"
                      name="managerName"
                      className="flex-1 flex-grow"
                      value={values.managerName}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      error={touched.managerName && Boolean(errors.managerName)}
                      helperText={touched.managerName && errors.managerName}
                    />
                    <TextField
                      type="text"
                      label="관리자 전화번호"
                      name="managerPhoneNumber"
                      className="flex-1 flex-grow"
                      value={values.managerPhoneNumber}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      error={
                        touched.managerPhoneNumber &&
                        Boolean(errors.managerPhoneNumber)
                      }
                      helperText={
                        touched.managerPhoneNumber && errors.managerPhoneNumber
                      }
                    />
                  </div>
                </div>
                <div className="flex flex-col">
                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={
                      isSubmitting || isCreateMemberPending || !idAvailable
                    }
                  >
                    회원가입
                  </Button>
                </div>
              </div>
            </div>
            <HospitalSearchModal
              open={openHospitalSearchModal}
              setOpen={setOpenHospitalSearchModal}
              setFieldValue={setFieldValue}
            />
          </Form>
        )}
      </Formik>
    </div>
  );
};

export default SignUp;
