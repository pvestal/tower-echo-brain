#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard for Echo Brain Model Routing
Interactive dashboard using Dash for visualization
"""

import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import psycopg2
import json
from datetime import datetime, timedelta
import numpy as np

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025'
}


class ModelRoutingDashboard:
    """Real-time dashboard for model routing monitoring"""

    def __init__(self, db_config=DB_CONFIG):
        self.db_config = db_config
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1('🧠 Echo Brain Model Routing Dashboard',
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.H3('Real-Time Performance Monitoring',
                       style={'textAlign': 'center', 'color': '#7f8c8d'})
            ]),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5000,  # Update every 5 seconds
                n_intervals=0
            ),

            # Top metrics cards
            html.Div([
                html.Div([
                    html.Div(id='total-queries-card', className='metric-card'),
                    html.Div(id='avg-satisfaction-card', className='metric-card'),
                    html.Div(id='avg-response-time-card', className='metric-card'),
                    html.Div(id='success-rate-card', className='metric-card'),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'})
            ]),

            # Main charts row
            html.Div([
                html.Div([
                    dcc.Graph(id='model-distribution-pie'),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(id='complexity-distribution'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),

            # Performance over time
            html.Div([
                dcc.Graph(id='performance-timeline'),
            ], style={'marginTop': '20px'}),

            # Model performance comparison
            html.Div([
                html.Div([
                    dcc.Graph(id='model-comparison-chart'),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(id='intent-heatmap'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={'marginTop': '20px'}),

            # Recent queries table
            html.Div([
                html.H3('Recent Queries', style={'textAlign': 'center'}),
                html.Div(id='recent-queries-table')
            ], style={'marginTop': '20px'}),

            # Active learning adjustments
            html.Div([
                html.H3('Active Learning Adjustments', style={'textAlign': 'center'}),
                html.Div(id='learning-adjustments-table')
            ], style={'marginTop': '20px'})

        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})

        # Add CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .metric-card {
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                        width: 200px;
                    }
                    .metric-value {
                        font-size: 32px;
                        font-weight: bold;
                        color: #2c3e50;
                    }
                    .metric-label {
                        font-size: 14px;
                        color: #7f8c8d;
                        margin-top: 5px;
                    }
                    .status-good { color: #27ae60; }
                    .status-warning { color: #f39c12; }
                    .status-bad { color: #e74c3c; }
                    table {
                        width: 100%;
                        background: white;
                        border-radius: 10px;
                        overflow: hidden;
                    }
                    th {
                        background: #34495e;
                        color: white;
                        padding: 10px;
                    }
                    td {
                        padding: 8px;
                        border-bottom: 1px solid #ecf0f1;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

    def _setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates"""

        @self.app.callback(
            [Output('total-queries-card', 'children'),
             Output('avg-satisfaction-card', 'children'),
             Output('avg-response-time-card', 'children'),
             Output('success-rate-card', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_metrics(n):
            """Update top metric cards"""
            metrics = self._get_current_metrics()

            total_card = html.Div([
                html.Div(f"{metrics['total_queries']:,}", className='metric-value'),
                html.Div('Total Queries (24h)', className='metric-label')
            ])

            satisfaction_class = 'status-good' if metrics['avg_satisfaction'] >= 4 else \
                               'status-warning' if metrics['avg_satisfaction'] >= 3 else 'status-bad'
            satisfaction_card = html.Div([
                html.Div(f"{metrics['avg_satisfaction']:.2f}",
                        className=f'metric-value {satisfaction_class}'),
                html.Div('Avg Satisfaction', className='metric-label')
            ])

            response_class = 'status-good' if metrics['avg_response_time'] < 1000 else \
                           'status-warning' if metrics['avg_response_time'] < 2000 else 'status-bad'
            response_card = html.Div([
                html.Div(f"{metrics['avg_response_time']:.0f}ms",
                        className=f'metric-value {response_class}'),
                html.Div('Avg Response Time', className='metric-label')
            ])

            success_class = 'status-good' if metrics['success_rate'] >= 0.9 else \
                          'status-warning' if metrics['success_rate'] >= 0.7 else 'status-bad'
            success_card = html.Div([
                html.Div(f"{metrics['success_rate']:.1%}",
                        className=f'metric-value {success_class}'),
                html.Div('Success Rate', className='metric-label')
            ])

            return total_card, satisfaction_card, response_card, success_card

        @self.app.callback(
            Output('model-distribution-pie', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_model_distribution(n):
            """Update model distribution pie chart"""
            data = self._get_model_distribution()

            fig = go.Figure(data=[go.Pie(
                labels=data['models'],
                values=data['counts'],
                hole=0.3,
                marker_colors=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
            )])

            fig.update_layout(
                title='Model Usage Distribution',
                height=300,
                showlegend=True
            )

            return fig

        @self.app.callback(
            Output('complexity-distribution', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_complexity_distribution(n):
            """Update complexity distribution histogram"""
            data = self._get_complexity_distribution()

            fig = go.Figure(data=[go.Histogram(
                x=data,
                nbinsx=20,
                marker_color='#3498db'
            )])

            fig.update_layout(
                title='Query Complexity Distribution',
                xaxis_title='Complexity Score',
                yaxis_title='Count',
                height=300
            )

            return fig

        @self.app.callback(
            Output('performance-timeline', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_performance_timeline(n):
            """Update performance timeline chart"""
            data = self._get_performance_timeline()

            fig = go.Figure()

            for model in data['models']:
                model_data = data['data'][model]
                fig.add_trace(go.Scatter(
                    x=model_data['timestamps'],
                    y=model_data['satisfaction'],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=2)
                ))

            fig.update_layout(
                title='Model Performance Over Time (24h)',
                xaxis_title='Time',
                yaxis_title='Average Satisfaction',
                height=400,
                hovermode='x unified'
            )

            return fig

        @self.app.callback(
            Output('model-comparison-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_model_comparison(n):
            """Update model comparison chart"""
            data = self._get_model_comparison()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Avg Response Time (ms)',
                x=data['models'],
                y=data['response_times'],
                yaxis='y',
                marker_color='#3498db'
            ))

            fig.add_trace(go.Scatter(
                name='Satisfaction Score',
                x=data['models'],
                y=data['satisfaction_scores'],
                yaxis='y2',
                mode='lines+markers',
                marker_color='#2ecc71',
                line=dict(width=3)
            ))

            fig.update_layout(
                title='Model Performance Comparison',
                yaxis=dict(title='Response Time (ms)', side='left'),
                yaxis2=dict(title='Satisfaction Score', overlaying='y', side='right'),
                height=400
            )

            return fig

        @self.app.callback(
            Output('intent-heatmap', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_intent_heatmap(n):
            """Update intent-model performance heatmap"""
            data = self._get_intent_model_matrix()

            fig = go.Figure(data=go.Heatmap(
                z=data['values'],
                x=data['models'],
                y=data['intents'],
                colorscale='RdYlGn',
                text=data['text'],
                texttemplate='%{text}',
                textfont={"size": 10}
            ))

            fig.update_layout(
                title='Intent-Model Performance Matrix',
                xaxis_title='Models',
                yaxis_title='Intents',
                height=400
            )

            return fig

        @self.app.callback(
            Output('recent-queries-table', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_recent_queries(n):
            """Update recent queries table"""
            queries = self._get_recent_queries()

            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th('Time'),
                        html.Th('Query'),
                        html.Th('Model'),
                        html.Th('Complexity'),
                        html.Th('Response (ms)'),
                        html.Th('Status')
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(q['time']),
                        html.Td(q['query'][:50] + '...' if len(q['query']) > 50 else q['query']),
                        html.Td(q['model']),
                        html.Td(q['complexity']),
                        html.Td(q['response_time']),
                        html.Td(q['status'], style={'color': self._get_status_color(q['status'])})
                    ]) for q in queries
                ])
            ])

            return table

        @self.app.callback(
            Output('learning-adjustments-table', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_learning_adjustments(n):
            """Update learning adjustments table"""
            adjustments = self._get_learning_adjustments()

            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th('Time'),
                        html.Th('Type'),
                        html.Th('Parameter'),
                        html.Th('Adjustment'),
                        html.Th('Confidence'),
                        html.Th('Applied')
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(a['time']),
                        html.Td(a['type']),
                        html.Td(a['parameter']),
                        html.Td(a['adjustment']),
                        html.Td(f"{a['confidence']:.1%}"),
                        html.Td('✓' if a['applied'] else '○',
                               style={'color': '#27ae60' if a['applied'] else '#95a5a6'})
                    ]) for a in adjustments
                ])
            ])

            return table

    def _get_current_metrics(self):
        """Get current performance metrics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction ELSE 0 END) as avg_satisfaction,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(CASE WHEN was_successful THEN 1 END)::FLOAT / COUNT(*) as success_rate
                FROM model_performance_log
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)

            result = cursor.fetchone()
            cursor.close()
            conn.close()

            return {
                'total_queries': result[0] or 0,
                'avg_satisfaction': result[1] or 0,
                'avg_response_time': result[2] or 0,
                'success_rate': result[3] or 0
            }
        except:
            return {
                'total_queries': 0,
                'avg_satisfaction': 0,
                'avg_response_time': 0,
                'success_rate': 0
            }

    def _get_model_distribution(self):
        """Get model usage distribution"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT model_used, COUNT(*) as count
                FROM model_performance_log
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY model_used
                ORDER BY count DESC
            """)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return {
                'models': [r[0] for r in results],
                'counts': [r[1] for r in results]
            }
        except:
            return {'models': [], 'counts': []}

    def _get_complexity_distribution(self):
        """Get complexity score distribution"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT complexity_score
                FROM model_performance_log
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                AND complexity_score IS NOT NULL
            """)

            results = [r[0] for r in cursor.fetchall()]
            cursor.close()
            conn.close()

            return results
        except:
            return []

    def _get_performance_timeline(self):
        """Get performance timeline data"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    DATE_TRUNC('hour', created_at) as hour,
                    model_used,
                    AVG(CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction ELSE 0 END) as avg_satisfaction
                FROM model_performance_log
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY hour, model_used
                ORDER BY hour
            """)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            data = {'models': [], 'data': {}}
            for row in results:
                model = row[1]
                if model not in data['models']:
                    data['models'].append(model)
                    data['data'][model] = {'timestamps': [], 'satisfaction': []}

                data['data'][model]['timestamps'].append(row[0])
                data['data'][model]['satisfaction'].append(row[2])

            return data
        except:
            return {'models': [], 'data': {}}

    def _get_model_comparison(self):
        """Get model comparison data"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    model_used,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction ELSE 0 END) as avg_satisfaction
                FROM model_performance_log
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY model_used
                ORDER BY avg_satisfaction DESC
            """)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return {
                'models': [r[0] for r in results],
                'response_times': [r[1] for r in results],
                'satisfaction_scores': [r[2] for r in results]
            }
        except:
            return {'models': [], 'response_times': [], 'satisfaction_scores': []}

    def _get_intent_model_matrix(self):
        """Get intent-model performance matrix"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    intent_type,
                    model_used,
                    AVG(CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction ELSE 0 END) as avg_satisfaction,
                    COUNT(*) as count
                FROM model_performance_log
                WHERE created_at >= NOW() - INTERVAL '7 days'
                AND intent_type IS NOT NULL
                GROUP BY intent_type, model_used
            """)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            # Build matrix
            intents = sorted(list(set([r[0] for r in results])))
            models = sorted(list(set([r[1] for r in results])))

            matrix = [[0 for _ in models] for _ in intents]
            text = [["" for _ in models] for _ in intents]

            for row in results:
                intent_idx = intents.index(row[0])
                model_idx = models.index(row[1])
                matrix[intent_idx][model_idx] = row[2]
                text[intent_idx][model_idx] = f"{row[2]:.2f}<br>n={row[3]}"

            return {
                'intents': intents,
                'models': models,
                'values': matrix,
                'text': text
            }
        except:
            return {'intents': [], 'models': [], 'values': [], 'text': []}

    def _get_recent_queries(self, limit=10):
        """Get recent queries"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    created_at,
                    query_text,
                    model_used,
                    complexity_score,
                    response_time_ms,
                    was_successful
                FROM model_performance_log
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return [
                {
                    'time': row[0].strftime('%H:%M:%S'),
                    'query': row[1] or 'N/A',
                    'model': row[2],
                    'complexity': row[3] or 0,
                    'response_time': f"{row[4]}ms" if row[4] else 'N/A',
                    'status': 'Success' if row[5] else 'Failed'
                }
                for row in results
            ]
        except:
            return []

    def _get_learning_adjustments(self, limit=5):
        """Get recent learning adjustments"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    created_at,
                    adjustment_type,
                    parameter_name,
                    new_value,
                    confidence,
                    applied
                FROM learning_adjustments
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return [
                {
                    'time': row[0].strftime('%H:%M:%S'),
                    'type': row[1],
                    'parameter': row[2],
                    'adjustment': str(row[3])[:30],
                    'confidence': row[4] or 0,
                    'applied': row[5]
                }
                for row in results
            ]
        except:
            return []

    def _get_status_color(self, status):
        """Get color for status"""
        return '#27ae60' if status == 'Success' else '#e74c3c'

    def run(self, debug=False, host='0.0.0.0', port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, host=host, port=port)


if __name__ == "__main__":
    dashboard = ModelRoutingDashboard()
    print("🚀 Starting Echo Brain Monitoring Dashboard...")
    print("📊 Dashboard available at: http://localhost:8050")
    dashboard.run(debug=True)