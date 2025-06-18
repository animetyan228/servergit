import React, { useState } from 'react';
import { Container, Form, Button, Spinner, Alert } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResponse(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResponse(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch("http://45.12.134.146:8080", {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setResponse(data);
      } else {
        setError(data?.ошибка || 'Ошибка сервера');
      }
    } catch (err) {
      setError('Ошибка подключения к серверу');
    }

    setLoading(false);
  };

  return (
    <Container className="mt-5">
      <h2 className="mb-4">Загрузка доверенности</h2>
      <Form.Group controlId="formFile" className="mb-3">
        <Form.Control type="file" accept=".jpg,.jpeg,.png,.pdf" onChange={handleFileChange} />
      </Form.Group>
      <Button onClick={handleUpload} disabled={loading}>
        {loading ? (
          <>
            <Spinner animation="border" size="sm" /> Загружаем...
          </>
        ) : (
          'Отправить'
        )}
      </Button>

      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}

      {response && (
        <div className="mt-4">
          <h4>Распознанный текст:</h4>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{response.распознанный_текст}</pre>
          <h4 className="mt-4">Структура:</h4>
          <pre>{JSON.stringify(response.структура, null, 2)}</pre>
        </div>
      )}
    </Container>
  );
}

export default App;
