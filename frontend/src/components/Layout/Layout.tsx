import { Outlet } from 'react-router-dom';

import Container from '@components/Layout/Container/Container';
import Footer from '@components/Layout/Footer/Footer';
import Header from '@components/Layout/Header/Header';

const Layout = () => {
  return (
    <div>
      <Header />
      <Container>
        <Outlet />
      </Container>
      <Footer />
    </div>
  );
};

export default Layout;
