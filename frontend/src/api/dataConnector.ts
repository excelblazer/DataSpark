import axios from 'axios';

export class DataConnectorApi {
    private baseUrl: string;
    
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }
    
    async fetchData(connectorType: string, query: any): Promise<any> {
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/connectors/${connectorType}/query`,
                query
            );
            return response.data;
        } catch (error) {
            console.error('Error fetching data:', error);
            throw error;
        }
    }
    
    async getSchema(connectorType: string, connectorId: string): Promise<any> {
        try {
            const response = await axios.get(
                `${this.baseUrl}/api/connectors/${connectorType}/${connectorId}/schema`
            );
            return response.data;
        } catch (error) {
            console.error('Error fetching schema:', error);
            throw error;
        }
    }
}
