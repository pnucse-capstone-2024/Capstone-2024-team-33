import { useAppDispatch } from '@lib/hooks/redux';
import { Box, Button, Divider, TextField } from '@mui/material';
import { login } from '@store/member.slice';
import { Form, Formik, FormikHelpers } from 'formik';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { useLogin } from '~/lib/hooks/useApi';

interface LoginInfo {
  id: string;
  password: string;
}

const initialValues: LoginInfo = {
  id: '',
  password: '',
};

const Login = () => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const { mutate: getMemberAuth } = useLogin();

  const handleSubmit = (
    values: LoginInfo,
    { setSubmitting }: FormikHelpers<LoginInfo>,
  ) => {
    getMemberAuth(
      { loginId: values.id, loginPassword: values.password },
      {
        onSuccess: (data) => {
          dispatch(
            login({
              name: data?.name ?? '',
              memberId: data?.id ?? -1,
              hospitalId: data?.hospitalId ?? [],
              role: data?.memberRole ?? 'ROLE_ANONYMOUS',
              token: data?.accessToken ?? '',
            }),
          );
          setSubmitting(false);
          setTimeout(() => {
            navigate('/');
          }, 0);
        },
        onError: () => {
          toast.error('아이디와 비밀번호를 다시 확인해주십시오.');
          setSubmitting(false);
        },
      },
    );
  };

  return (
    <div className="flex h-full items-center justify-center">
      <Formik initialValues={initialValues} onSubmit={handleSubmit}>
        {({ isSubmitting, values, handleChange, handleBlur }) => (
          <Form className="flex w-full max-w-sign flex-col gap-2 rounded-lg bg-white px-10 py-16 shadow-md">
            <div className="flex flex-col gap-2">
              <Box className="py-4">
                <p className="mb-1 text-xl">로그인</p>
                <Divider />
              </Box>

              <TextField
                type="text"
                name="id"
                label="ID"
                value={values.id}
                onChange={handleChange}
                onBlur={handleBlur}
              />
              <TextField
                type="password"
                name="password"
                label="Password"
                value={values.password}
                onChange={handleChange}
                onBlur={handleBlur}
              />

              <Button
                type="submit"
                variant="contained"
                size="large"
                disabled={isSubmitting}
                sx={{
                  marginTop: '16px',
                }}
              >
                로그인
              </Button>
            </div>
            <div className="flex justify-center gap-6 text-primary">
              Safe Catcher에 등록되어 있지 않나요?
              <Link to="/sign-up" className="font-bold underline">
                회원가입하기
              </Link>
            </div>
          </Form>
        )}
      </Formik>
    </div>
  );
};

export default Login;
