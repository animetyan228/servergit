import React, { useState, useEffect } from 'react';
import { Container, Form, Button, Spinner, Alert } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const BACKEND_URL = "http://45.12.134.146:8080";

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${BACKEND_URL}/`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (res.ok) {
        setPolling(true);
      } else {
        setError(data.ошибка || "Ошибка сервера");
      }
    } catch (err) {
      setError("Ошибка подключения к серверу");
    }

    setLoading(false);
  };

  useEffect(() => {
    let interval;
    if (polling) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${BACKEND_URL}/result`);
          const data = await res.json();

          if (data.status === "SUCCESS") {
            setResult(data.результат);
            setPolling(false);
          } else if (data.status === "ошибка") {
            setError(data.детали || "Неизвестная ошибка");
            setPolling(false);
          }
        } catch {
          setError("Ошибка получения результата");
          setPolling(false);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [polling]);

  return (
    <Container className="mt-5">
      <h2 className="mb-4">Загрузка доверенности</h2>
      <Form.Group controlId="formFile" className="mb-3">
        <Form.Control type="file" accept=".jpg,.jpeg,.png,.pdf" onChange={handleFileChange} />
      </Form.Group>
      <Button onClick={handleUpload} disabled={loading || polling}>
        {loading ? <><Spinner size="sm" animation="border" /> Отправка...</> : "Отправить"}
      </Button>

      {polling && <Alert className="mt-3" variant="info">Идёт обработка...</Alert>}
      {error && <Alert className="mt-3" variant="danger">{error}</Alert>}

      {result && (
        <div className="mt-4">
          <h5>Распознанный текст:</h5>
          <pre>{result.распознанный_текст}</pre>
          <h5 className="mt-3">Структура:</h5>
          <pre>{JSON.stringify(result.структура, null, 2)}</pre>
        </div>
      )}
    </Container>
  );
}

export default App;
