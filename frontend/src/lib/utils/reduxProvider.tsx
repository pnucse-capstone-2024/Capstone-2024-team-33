import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';

import { persistor, store } from '@store/index';

type Props = {
  children: React.ReactNode;
};

const ReduxProvider = ({ children }: Props) => (
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
      {children}
    </PersistGate>
  </Provider>
);
export default ReduxProvider;
